import torch
import triton
import triton.language as tl


@triton.jit
def backward_dx_kernel(
    Grad_Vals_ptr, Indices_ptr, W_ptr,
    Grad_X_ptr,
    TotalTokens, D, V, K: tl.constexpr,
    stride_gt, stride_gk,
    stride_it, stride_ik,
    stride_wv, stride_wd,
    stride_gxt, stride_gxd,
    BLOCK_D: tl.constexpr
):
    pid = tl.program_id(0)
    row_idx = pid
    
    # Accumulator for the gradient of the specific token vector
    grad_x_acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    
    # Pointers to specific row for this token
    g_vals_ptr = Grad_Vals_ptr + row_idx * stride_gt
    inds_ptr = Indices_ptr + row_idx * stride_it
    
    # Iterate over the K selected indices
    for k in range(K):
        idx_v = tl.load(inds_ptr + k * stride_ik)
        g_val = tl.load(g_vals_ptr + k * stride_gk).to(tl.float32) 
        
        # Load the weight vector corresponding to the selected index
        w_ptr = W_ptr + idx_v * stride_wv + tl.arange(0, BLOCK_D) * stride_wd
        mask_d = tl.arange(0, BLOCK_D) < D
        w_row = tl.load(w_ptr, mask=mask_d, other=0.0).to(tl.float32) 
        
        grad_x_acc += g_val * w_row
        
    # Store dX
    out_ptr = Grad_X_ptr + row_idx * stride_gxt + tl.arange(0, BLOCK_D) * stride_gxd
    mask_d = tl.arange(0, BLOCK_D) < D
    tl.store(out_ptr, grad_x_acc, mask=mask_d)

class SparseColbert(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, k):
        # x: [B, T, D]
        # w: [V, D]
        B, T, D = x.shape
        V, _ = w.shape
        
        # Flatten B and T into a single dimension of total tokens
        # effectively [B*T, D]
        x_flat = x.view(-1, D)
        total_tokens = B * T
        
        # Pre-allocate outputs in flattened shape
        top_vals_flat = torch.empty((total_tokens, k), device=x.device, dtype=x.dtype)
        top_inds_flat = torch.empty((total_tokens, k), device=x.device, dtype=torch.long)
        
        # Chunked Forward to keep memory low
        CHUNK_SIZE = 1024
        for i in range(0, total_tokens, CHUNK_SIZE):
            end = min(i + CHUNK_SIZE, total_tokens)
            x_chunk = x_flat[i:end]
            
            # Matmul: [Chunk, D] @ [D, V] -> [Chunk, V]
            logits_chunk = torch.matmul(x_chunk, w.t())
            
            vals_c, inds_c = torch.topk(logits_chunk, k, dim=1)
            top_vals_flat[i:end] = vals_c
            top_inds_flat[i:end] = inds_c

        # Reshape outputs back to [B, T, K]
        top_vals = top_vals_flat.view(B, T, k)
        top_inds = top_inds_flat.view(B, T, k)

        ctx.save_for_backward(x, w, top_inds)
        ctx.k = k
        return top_vals, top_inds

    @staticmethod
    def backward(ctx, grad_vals, _grad_inds):
        # grad_vals: [B, T, K]
        x, w, top_inds = ctx.saved_tensors
        B, T, D = x.shape
        V, _ = w.shape
        
        # Flatten tensors for processing as a list of tokens
        # We ensure contiguous memory for correct pointer math in Triton
        grad_vals_flat = grad_vals.view(-1, ctx.k).contiguous() # [B*T, K]
        top_inds_flat = top_inds.view(-1, ctx.k).contiguous()   # [B*T, K]
        x_flat = x.view(-1, D).contiguous()                     # [B*T, D]
        w = w.contiguous()
        
        total_tokens = B * T
        
        # 1. dL/dX (Triton Kernel)
        grad_x_flat = torch.empty_like(x_flat)
        
        grid = (total_tokens, )
        BLOCK_D = triton.next_power_of_2(D)
        
        backward_dx_kernel[grid](
            grad_vals_flat, top_inds_flat, w,
            grad_x_flat,
            total_tokens, D, V, ctx.k,
            grad_vals_flat.stride(0), grad_vals_flat.stride(1),
            top_inds_flat.stride(0), top_inds_flat.stride(1),
            w.stride(0), w.stride(1),
            grad_x_flat.stride(0), grad_x_flat.stride(1),
            BLOCK_D=BLOCK_D
        )
        
        # Reshape dX back to [B, T, D]
        grad_x = grad_x_flat.view(B, T, D)
        
        # 2. dL/dW (Chunked Scatter Add)
        grad_w = torch.zeros_like(w)
        
        SCATTER_BATCH = 256 
        
        for start in range(0, total_tokens, SCATTER_BATCH):
            end = min(start + SCATTER_BATCH, total_tokens)
            
            # Slice flattened tensors
            t_slice = top_inds_flat[start:end]    # [Batch, K]
            g_slice = grad_vals_flat[start:end]   # [Batch, K]
            x_slice = x_flat[start:end]           # [Batch, D]
            
            # Flatten indices and gradients for the scatter op
            flat_inds = t_slice.reshape(-1)       # [Batch*K]
            flat_grads = g_slice.reshape(-1)      # [Batch*K]
            
            # Expand X: [Batch, D] -> [Batch, K, D] -> [Batch*K, D]
            # This replicates the input vector K times, one for each selected logit
            x_expanded = x_slice.unsqueeze(1).expand(-1, ctx.k, -1).reshape(-1, D)
            
            # Scale input by the incoming gradient
            scaled_x = x_expanded * flat_grads.unsqueeze(1)
            
            # Accumulate into global weights
            grad_w.index_add_(0, flat_inds, scaled_x)
        
        return grad_x, grad_w, None

def sparse_colbert_topk(x, w, k):
    return SparseColbert.apply(x, w, k)

