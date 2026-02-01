import triton
import triton.language as tl
import time
import torch



@triton.jit
def splade_forward_kernel_fused(
    X_ptr, W_ptr, Out_Vals_ptr, Out_Inds_ptr,
    B, T, V, D,
    stride_xb, stride_xt, stride_xd,
    stride_wv, stride_wd,
    stride_ob, stride_ov,
    BLOCK_T: tl.constexpr, BLOCK_V: tl.constexpr, BLOCK_D: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr, ALLOW_TF32: tl.constexpr
):
    # L2 Cache Swizzling to maximize W reuse
    pid = tl.program_id(axis=0)
    num_pid_v = tl.cdiv(V, BLOCK_V)
    num_pid_b = B
    num_pid_in_group = GROUP_SIZE_M * num_pid_v
    group_id = pid // num_pid_in_group
    first_pid_b = group_id * GROUP_SIZE_M
    group_size_b = min(num_pid_b - first_pid_b, GROUP_SIZE_M)
    
    pid_b = first_pid_b + (pid % group_size_b)
    pid_v = (pid % num_pid_in_group) // group_size_b
    
    offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    
    # Init accumulators
    # if the initial max val is set to -inf, it can easily lead to numerical errors. 
    # Since relu will be applied to the outputs anyway, we can set it to 0 to remove any negative values.
    acc_max_val = tl.full([BLOCK_V], value=0, dtype=tl.float32)
    acc_max_ind = tl.full([BLOCK_V], value=0, dtype=tl.int32)
    
    # Iterate over T in blocks
    for t_start in range(0, T, BLOCK_T):
        dot_acc = tl.zeros([BLOCK_T, BLOCK_V], dtype=tl.float32)
        
        # Iterate over D (Inner product)
        for d_start in range(0, D, BLOCK_D):
            offs_d = d_start + tl.arange(0, BLOCK_D)
            mask_d = offs_d < D
            
            # Load X [BLOCK_T, BLOCK_D]
            offs_t = t_start + tl.arange(0, BLOCK_T)
            mask_t = offs_t < T
            x_ptr = X_ptr + pid_b * stride_xb + (offs_t[:, None] * stride_xt + offs_d[None, :] * stride_xd)
            x_tile = tl.load(x_ptr, mask=mask_t[:, None] & mask_d[None, :], other=0.0)
            
            # Load W [BLOCK_V, BLOCK_D]
            w_ptr = W_ptr + (offs_v[:, None] * stride_wv + offs_d[None, :] * stride_wd)
            mask_v = offs_v < V
            w_tile = tl.load(w_ptr, mask=mask_v[:, None] & mask_d[None, :], other=0.0)
            
            # Fused Matmul
            dot_acc += tl.dot(x_tile, w_tile.trans(), allow_tf32=ALLOW_TF32)

        # Local Max over T
        offs_t = t_start + tl.arange(0, BLOCK_T)
        mask_t = offs_t < T
        dot_acc = tl.where(mask_t[:, None], dot_acc, 0.0)
        
        local_max_val = tl.max(dot_acc, axis=0)
        local_argmax_rel = tl.argmax(dot_acc, axis=0)
        local_max_ind = t_start + local_argmax_rel
        
        # Global Max Update
        is_new_max = local_max_val > acc_max_val
        acc_max_val = tl.where(is_new_max, local_max_val, acc_max_val)
        acc_max_ind = tl.where(is_new_max, local_max_ind, acc_max_ind)
        
    # Store
    mask_v = offs_v < V
    out_val_ptr = Out_Vals_ptr + pid_b * stride_ob + offs_v * stride_ov
    out_ind_ptr = Out_Inds_ptr + pid_b * stride_ob + offs_v * stride_ov
    tl.store(out_val_ptr, acc_max_val, mask=mask_v)
    tl.store(out_ind_ptr, acc_max_ind, mask=mask_v)

@triton.jit
def splade_backward_kernel(
    Grad_Y_ptr, Indices_ptr, X_ptr, W_ptr,
    Grad_X_ptr, Grad_W_ptr,
    B, T, V, D,
    stride_gy_b, stride_gy_v,
    stride_ind_b, stride_ind_v,
    stride_xb, stride_xt, stride_xd,
    stride_wv, stride_wd,
    stride_gx_b, stride_gx_t, stride_gx_d,
    stride_gw_v, stride_gw_d,
    BLOCK_V: tl.constexpr, BLOCK_D: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_v = tl.program_id(1)
    v_start = pid_v * BLOCK_V
    mask_d = tl.arange(0, BLOCK_D) < D
    
    # Iterate serially over V to handle unique T indices safely
    for k in range(BLOCK_V):
        current_v = v_start + k
        if current_v < V:
            # Load Gradient dY
            gy_ptr = Grad_Y_ptr + pid_b * stride_gy_b + current_v * stride_gy_v
            grad_val = tl.load(gy_ptr).to(tl.float32)
            
            if grad_val != 0.0:
                # Get the T index selected
                ind_ptr = Indices_ptr + pid_b * stride_ind_b + current_v * stride_ind_v
                t_idx = tl.load(ind_ptr)
                
                # Load W[v] and X[b, t]
                w_vec = tl.load(W_ptr + current_v * stride_wv + tl.arange(0, BLOCK_D) * stride_wd, mask=mask_d).to(tl.float32)
                x_vec = tl.load(X_ptr + pid_b * stride_xb + t_idx * stride_xt + tl.arange(0, BLOCK_D) * stride_xd, mask=mask_d).to(tl.float32)
                
                # Atomic Scatter Add
                # dX[b, t] += g * W[v]
                tl.atomic_add(Grad_X_ptr + pid_b * stride_gx_b + t_idx * stride_gx_t + tl.arange(0, BLOCK_D) * stride_gx_d, w_vec * grad_val, mask=mask_d)
                # dW[v] += g * X[b, t]
                tl.atomic_add(Grad_W_ptr + current_v * stride_gw_v + tl.arange(0, BLOCK_D) * stride_gw_d, x_vec * grad_val, mask=mask_d)

class SpladeMaxPool(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w):
        B, T, D = x.shape
        V, _ = w.shape
        x = x.contiguous(); w = w.contiguous()
        out_vals = torch.empty((B, V), device=x.device, dtype=x.dtype)
        out_inds = torch.empty((B, V), device=x.device, dtype=torch.int32)
        
        # --- TUNED CONFIGURATION ---
        BLOCK_T = 32
        BLOCK_V = 128
        BLOCK_D = 64
        GROUP_SIZE_M = 8
        ALLOW_TF32 = False # Strict precision
        # Stages=2 prevents SRAM OOM on smaller cards while keeping BLOCK_V large
        NUM_STAGES = 2 
        NUM_WARPS = 4

        total_programs = B * triton.cdiv(V, BLOCK_V)
        
        splade_forward_kernel_fused[(total_programs,)](
            x, w, out_vals, out_inds,
            B, T, V, D,
            x.stride(0), x.stride(1), x.stride(2),
            w.stride(0), w.stride(1),
            out_vals.stride(0), out_vals.stride(1),
            BLOCK_T=BLOCK_T, BLOCK_V=BLOCK_V, BLOCK_D=BLOCK_D,
            GROUP_SIZE_M=GROUP_SIZE_M, ALLOW_TF32=ALLOW_TF32,
            num_warps=NUM_WARPS, num_stages=NUM_STAGES
        )
        ctx.save_for_backward(x, w, out_inds)
        return out_vals

    @staticmethod
    def backward(ctx, grad_output):
        x, w, out_inds = ctx.saved_tensors
        B, T, D = x.shape
        V, _ = w.shape
        grad_output = grad_output.contiguous()
        
        # Accumulate in Float32 for stability
        dx = torch.zeros((B, T, D), device=x.device, dtype=torch.float32)
        dw = torch.zeros((V, D), device=x.device, dtype=torch.float32)
        
        BLOCK_V = 64
        BLOCK_D = triton.next_power_of_2(D)
        grid = (B, triton.cdiv(V, BLOCK_V))
        
        splade_backward_kernel[grid](
            grad_output, out_inds, x, w,
            dx, dw,
            B, T, V, D,
            grad_output.stride(0), grad_output.stride(1),
            out_inds.stride(0), out_inds.stride(1),
            x.stride(0), x.stride(1), x.stride(2),
            w.stride(0), w.stride(1),
            dx.stride(0), dx.stride(1), dx.stride(2),
            dw.stride(0), dw.stride(1),
            BLOCK_V=BLOCK_V, BLOCK_D=BLOCK_D,
            num_warps=4
        )
        return dx.to(x.dtype), dw.to(w.dtype)

def splade_max_pool(x, w):
    return SpladeMaxPool.apply(x, w)


