import torch
import triton
import triton.language as tl


def _sparse_maxsim(
    q_vals, q_inds, d_vals, d_inds, 
    chunk_b1=32, chunk_b2=32, 
    dtype=torch.bfloat16  # Force execution type
):
    """
    Computes pairwise MaxSim strictly in the specified precision (fp16/bf16).
    
    Optimizations:
    1. Projects sparse vectors to 'Active Query Subspace'.
    2. Pads subspace dimension to nearest multiple of 16 for Tensor Core alignment.
    3. Keeps all intermediate dense tensors in 'dtype'.
    """
    # 1. Cast Inputs to target dtype (View-only if already correct)
    q_vals = q_vals.to(dtype)
    d_vals = d_vals.to(dtype)
    
    B1, Tq, K = q_vals.shape
    B2, Td, _ = d_vals.shape
    device = q_vals.device
    
    # Output can be kept in dtype or float32. 
    # Usually float32 is safer for the final sum, but we respect your request.
    final_scores = torch.zeros((B1, B2), device=device, dtype=dtype)
    
    if q_inds.numel() == 0 or d_inds.numel() == 0:
        return final_scores
        
    V_max = max(q_inds.max().item(), d_inds.max().item()) + 1
    
    # Iterate Query Chunks
    for i in range(0, B1, chunk_b1):
        i_end = min(i + chunk_b1, B1)
        bs_q = i_end - i
        
        q_v_chunk = q_vals[i:i_end]
        q_i_chunk = q_inds[i:i_end]
        
        # --- A. Build Active Subspace ---
        active_vocab, _ = torch.unique(q_i_chunk.flatten(), sorted=True, return_inverse=True)
        raw_dim = active_vocab.size(0)
        
        if raw_dim == 0: continue
            
        # --- OPTIMIZATION: Tensor Core Alignment ---
        # Pad dimension to multiple of 16 (for fp16/bf16 HMMA units)
        remainder = raw_dim % 16
        padding = (16 - remainder) % 16
        padded_dim = raw_dim + padding
        
        # Build Mapping
        mapping = torch.full((V_max,), -1, dtype=torch.long, device=device)
        mapping[active_vocab.long()] = torch.arange(raw_dim, device=device)
        
        # --- B. Create Dense Q (Padded) ---
        q_i_mapped = mapping[q_i_chunk.long()]
        
        # Allocate in target DTYPE with PADDED dimension
        Q_dense = torch.zeros((bs_q, Tq, padded_dim), device=device, dtype=dtype)
        
        # Scatter values (only into valid spots, padding remains 0)
        Q_dense.scatter_(2, q_i_mapped, q_v_chunk)
        
        # Flatten: [bs_q*Tq, padded_dim]
        Q_flat = Q_dense.view(bs_q * Tq, padded_dim)
        
        # Iterate Doc Chunks
        for j in range(0, B2, chunk_b2):
            j_end = min(j + chunk_b2, B2)
            bs_d = j_end - j
            
            d_v_chunk = d_vals[j:j_end]
            d_i_chunk = d_inds[j:j_end]
            
            # --- C. Create Dense D (Padded) ---
            d_i_mapped = mapping[d_i_chunk.long()]
            mask = d_i_mapped >= 0
            
            D_dense = torch.zeros((bs_d, Td, padded_dim), device=device, dtype=dtype)
            
            # Scatter logic
            coords = torch.nonzero(mask, as_tuple=True)
            # Safe index_put_ into padded tensor
            D_dense.index_put_((coords[0], coords[1], d_i_mapped[mask]), d_v_chunk[mask])
            
            D_flat = D_dense.view(bs_d * Td, padded_dim)
            
            # --- D. MatMul (Strictly FP16/BF16) ---
            # Dimensions are [M, K] @ [K, N] where K=padded_dim (multiple of 16)
            # This guarantees Tensor Core usage.
            score_matrix = torch.matmul(Q_flat, D_flat.t())
            
            # --- E. Reduction ---
            score_matrix = score_matrix.view(bs_q, Tq, bs_d, Td).permute(0, 2, 1, 3)
            
            # Max/Sum in target dtype
            max_sims = score_matrix.max(dim=3).values
            block_scores = max_sims.sum(dim=2)
            
            final_scores[i:i_end, j:j_end] = block_scores
            
    return final_scores



@triton.jit
def _pairwise_maxsim_bwd_kernel_tiled(
    # Pointers
    Grad_Score_ptr, Argmax_ptr,
    Q_Vals_ptr, Q_Inds_ptr,
    D_Vals_ptr, D_Inds_ptr,
    DQ_Vals_ptr, DD_Vals_ptr,
    # Shapes
    B1, B2, Tq, Td, K,
    # Strides
    stride_gs_b1, stride_gs_b2,
    stride_a_b1, stride_a_b2, stride_a_t,
    stride_q_b, stride_q_t, stride_q_k,
    stride_d_b, stride_d_t, stride_d_k,
    stride_dq_b, stride_dq_t, stride_dq_k,
    stride_dd_b, stride_dd_t, stride_dd_k,
    # Constants
    BLOCK_K: tl.constexpr
):
    # Grid: (Tq, B1, B2)
    pid_tq = tl.program_id(0)
    pid_b1 = tl.program_id(1)
    pid_b2 = tl.program_id(2)

    # 1. Retrieve Winner Index
    idx_ptr = Argmax_ptr + pid_b1 * stride_a_b1 + pid_b2 * stride_a_b2 + pid_tq * stride_a_t
    best_idx = tl.load(idx_ptr)
    
    if best_idx < 0: return

    # 2. Load Gradient Scalar
    grad_val = tl.load(Grad_Score_ptr + pid_b1 * stride_gs_b1 + pid_b2 * stride_gs_b2).to(tl.float32)

    # 3. Base Pointers
    q_base_v = Q_Vals_ptr + pid_b1 * stride_q_b + pid_tq * stride_q_t
    q_base_i = Q_Inds_ptr + pid_b1 * stride_q_b + pid_tq * stride_q_t
    
    d_base_v = D_Vals_ptr + pid_b2 * stride_d_b + best_idx * stride_d_t
    d_base_i = D_Inds_ptr + pid_b2 * stride_d_b + best_idx * stride_d_t

    dq_base = DQ_Vals_ptr + pid_b1 * stride_dq_b + pid_tq * stride_dq_t
    dd_base = DD_Vals_ptr + pid_b2 * stride_dd_b + best_idx * stride_dd_t

    # 4. Tiled Loop (Handles K=512 safely)
    for k_q_off in range(0, K, BLOCK_K):
        offs_kq = k_q_off + tl.arange(0, BLOCK_K)
        mask_q = offs_kq < K
        
        # Load Q (Auto-cast to float32 for math)
        q_i = tl.load(q_base_i + offs_kq * stride_q_k, mask=mask_q, other=-1)
        q_v = tl.load(q_base_v + offs_kq * stride_q_k, mask=mask_q, other=0.0).to(tl.float32)
        
        dq_acc = tl.zeros([BLOCK_K], dtype=tl.float32)
        
        for k_d_off in range(0, K, BLOCK_K):
            offs_kd = k_d_off + tl.arange(0, BLOCK_K)
            mask_d = offs_kd < K
            
            # Load D
            d_i = tl.load(d_base_i + offs_kd * stride_d_k, mask=mask_d, other=-2)
            d_v = tl.load(d_base_v + offs_kd * stride_d_k, mask=mask_d, other=0.0).to(tl.float32)
            
            # Broadcast Compare
            match = (q_i[:, None] == d_i[None, :])
            
            # Accumulate dQ locally
            dq_acc += tl.sum(tl.where(match, d_v[None, :], 0.0), axis=1)
            
            # Accumulate dD (Atomic Add immediately)
            dd_partial = tl.sum(tl.where(match, q_v[:, None], 0.0), axis=0) * grad_val
            tl.atomic_add(dd_base + offs_kd * stride_dd_k, dd_partial, mask=mask_d)

        # Write dQ
        dq_acc = dq_acc * grad_val
        tl.atomic_add(dq_base + offs_kq * stride_dq_k, dq_acc, mask=mask_q)

def matmul_fp8(A, B, out_dtype=torch.float32):
    # Attempt FP8 Tensor Core call
    if hasattr(torch, "_scaled_mm") and A.dtype == torch.float8_e4m3fn:
        try:
            scale_a = torch.tensor(1.0, device=A.device, dtype=torch.float32)
            scale_b = torch.tensor(1.0, device=B.device, dtype=torch.float32)
            return torch._scaled_mm(A, B, scale_a, scale_b, out_dtype=out_dtype)
        except: pass
    # Fallback
    return torch.matmul(A.to(torch.bfloat16), B.to(torch.bfloat16))

class PairwiseMaxSimFp8(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q_vals, q_inds, d_vals, d_inds, chunk_size=4):
        device = q_vals.device
        dtype = q_vals.dtype
        B1, Tq, K = q_vals.shape
        B2, Td, _ = d_vals.shape

        scores = torch.zeros((B1, B2), device=device, dtype=torch.float32)
        argmax_idxs = torch.empty((B1, B2, Tq), device=device, dtype=torch.int32)
        
        if q_inds.numel() == 0 or d_inds.numel() == 0: return scores
        V_max = max(q_inds.max().item(), d_inds.max().item()) + 1

        # Check for FP8 Support
        use_fp8 = hasattr(torch, "float8_e4m3fn") and torch.cuda.get_device_capability()[0] >= 9
        fp8_dtype = torch.float8_e4m3fn if use_fp8 else dtype

        # --- Chunked Forward ---
        for i in range(0, B1, chunk_size):
            i_end = min(i + chunk_size, B1)
            bs_q = i_end - i
            q_v, q_i = q_vals[i:i_end], q_inds[i:i_end]

            # 1. Build Subspace
            all_inds = torch.cat([q_i.flatten(), d_inds.flatten()])
            active_vocab, _ = torch.unique(all_inds, sorted=True, return_inverse=True)
            subspace_dim = active_vocab.size(0)
            pad = (16 - (subspace_dim % 16)) % 16
            padded_dim = subspace_dim + pad
            mapping = torch.full((V_max,), -1, dtype=torch.int32, device=device)
            mapping[active_vocab.long()] = torch.arange(subspace_dim, device=device, dtype=torch.int32)
            
            # 2. Densify Q (Convert to FP8 here if available)
            Q_dense = torch.zeros((bs_q, Tq, padded_dim), device=device, dtype=fp8_dtype)
            q_mapped = mapping[q_i.long()].long()
            Q_dense.scatter_(2, q_mapped, q_v.to(fp8_dtype))
            Q_flat = Q_dense.view(bs_q * Tq, padded_dim)
            
            # 3. Densify D (Convert to FP8 here)
            d_mapped = mapping[d_inds.long()].long()
            mask_d = d_mapped >= 0
            D_dense = torch.zeros((B2, Td, padded_dim), device=device, dtype=fp8_dtype)
            coords_d = torch.nonzero(mask_d, as_tuple=True)
            D_dense.index_put_((coords_d[0], coords_d[1], d_mapped[mask_d]), d_vals[mask_d].to(fp8_dtype))
            D_flat = D_dense.view(B2 * Td, padded_dim)
            
            # 4. MatMul & Reduce
            score_mat = matmul_fp8(Q_flat, D_flat.t())
            score_mat = score_mat.view(bs_q, Tq, B2, Td).permute(0, 2, 1, 3)
            max_v, max_i = score_mat.max(dim=3)
            scores[i:i_end] = max_v.float().sum(dim=2)
            argmax_idxs[i:i_end] = max_i.to(torch.int32)

        # Save BF16 inputs for backward (Robustness > Memory saving here)
        ctx.save_for_backward(q_vals, q_inds, d_vals, d_inds, argmax_idxs)
        return scores

    @staticmethod
    def backward(ctx, grad_output):
        q_vals, q_inds, d_vals, d_inds, argmax_idxs = ctx.saved_tensors
        
        # Initialize Grads (Float32 for atomic safety in Triton)
        grad_q = torch.zeros_like(q_vals, dtype=torch.float32)
        grad_d = torch.zeros_like(d_vals, dtype=torch.float32)
        
        B1, Tq, K = q_vals.shape
        B2, Td, _ = d_vals.shape
        
        # Helper for Triton Stride
        def ensure_contig(x): return x.contiguous()
        
        # Launch Triton
        grid = (Tq, B1, B2)
        _pairwise_maxsim_bwd_kernel_tiled[grid](
            ensure_contig(grad_output), ensure_contig(argmax_idxs),
            ensure_contig(q_vals), ensure_contig(q_inds),
            ensure_contig(d_vals), ensure_contig(d_inds),
            grad_q, grad_d,
            B1, B2, Tq, Td, K,
            grad_output.stride(0), grad_output.stride(1),
            argmax_idxs.stride(0), argmax_idxs.stride(1), argmax_idxs.stride(2),
            q_vals.stride(0), q_vals.stride(1), q_vals.stride(2),
            d_vals.stride(0), d_vals.stride(1), d_vals.stride(2),
            grad_q.stride(0), grad_q.stride(1), grad_q.stride(2),
            grad_d.stride(0), grad_d.stride(1), grad_d.stride(2),
            BLOCK_K=64, num_warps=4
        )
        
        return grad_q.to(q_vals.dtype), None, grad_d.to(d_vals.dtype), None, None



def sparse_maxsim(q, d, q_mask=None, d_mask=None):
    """
    Computes pairwise MaxSim scores.
    Args:
        q_vals, q_inds: (B1, Tq, K)
        d_vals, d_inds: (B2, Td, K)
    Returns:
        scores: (B1, B2)
    """
    q_vals, q_inds = q
    d_vals, d_inds = d

    scores = PairwiseMaxSimFp8.apply(
    #scores = _sparse_maxsim(
        q_vals, q_inds,
        d_vals, d_inds,
    )

    if q_mask is not None:
        denom = q_mask.sum(dim=1).clamp_min(1.0)  # (B1,)
        scores = scores / denom[:, None]
    else:
        denom = q_vals.size(1)  # Tq
        scores = scores / denom

    return scores


def sparse_maxsim_pairwise(q, d, q_mask=None, d_mask=None, normalize=True):
    """
    Pairwise MaxSim with mean pooling over query tokens.

    Args:
        q: (q_vals, q_inds) each (B, Tq, K)
        d: (d_vals, d_inds) each (B, Td, K)
        q_mask: (B, Tq) optional

    Returns:
        scores: (B,)
    """
    q_vals, q_inds = q
    d_vals, d_inds = d

    B, Tq, _ = q_vals.shape
    scores = []

    for i in range(B):
        
        s = PairwiseMaxSimFp8.apply(
            q_vals[i:i+1], q_inds[i:i+1],
            d_vals[i:i+1], d_inds[i:i+1],
        )  # (1, 1)

        scores.append(s.squeeze())  # scalar

    scores = torch.stack(scores)  # (B,)

    # -------------------------------
    # Mean pooling over query tokens
    # -------------------------------
    if normalize == True:
        if q_mask is not None:
            denom = q_mask.sum(dim=1).clamp_min(1.0)  # (B,)
        else:
            denom = Tq

        scores = scores / denom
    return scores



