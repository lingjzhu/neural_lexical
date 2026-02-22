import torch
import triton
import triton.language as tl

@triton.jit
def index_add_kernel(
    w_ptr, cluster_ids_ptr, w_reduced_ptr,
    H: tl.constexpr, V: tl.constexpr, C: tl.constexpr,
    stride_w_v, stride_w_h,
    stride_r_c, stride_r_h,
    BLOCK_SIZE_V: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
):
    pid_v = tl.program_id(0)
    pid_h = tl.program_id(1)

    offs_v = pid_v * BLOCK_SIZE_V + tl.arange(0, BLOCK_SIZE_V)
    mask_v = offs_v < V

    offs_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    mask_h = offs_h < H

    # load clusters
    c_ids = tl.load(cluster_ids_ptr + offs_v, mask=mask_v, other=-1)
    
    # load W
    w_ptrs = w_ptr + (offs_v[:, None] * stride_w_v + offs_h[None, :] * stride_w_h)
    w_vals = tl.load(w_ptrs, mask=mask_v[:, None] & mask_h[None, :], other=0.0)

    # We only want to accumulate if c_id != -1
    valid_mask = (c_ids[:, None] != -1) & mask_h[None, :]
    r_ptrs = w_reduced_ptr + (c_ids[:, None] * stride_r_c + offs_h[None, :] * stride_r_h)
    
    tl.atomic_add(r_ptrs, w_vals, mask=valid_mask)

def fused_index_add(w: torch.Tensor, cluster_ids: torch.Tensor, num_clusters: int, dtype=None) -> torch.Tensor:
    if dtype is None:
        dtype = w.dtype
    V, H = w.shape
    assert cluster_ids.shape[0] == V

    w_reduced = torch.zeros((num_clusters, H), device=w.device, dtype=dtype)

    BLOCK_SIZE_V = 16
    BLOCK_SIZE_H = triton.next_power_of_2(H) if H <= 128 else 128
    
    grid = (
        triton.cdiv(V, BLOCK_SIZE_V),
        triton.cdiv(H, BLOCK_SIZE_H)
    )

    w = w.contiguous()
    cluster_ids = cluster_ids.contiguous()

    index_add_kernel[grid](
        w, cluster_ids, w_reduced,
        H, V, num_clusters,
        w.stride(0), w.stride(1),
        w_reduced.stride(0), w_reduced.stride(1),
        BLOCK_SIZE_V=BLOCK_SIZE_V,
        BLOCK_SIZE_H=BLOCK_SIZE_H
    )

    return w_reduced

@triton.jit
def fused_sim_argmax_kernel(
    w_ptr, w_reduced_ptr, new_cluster_ids_ptr,
    V: tl.constexpr, C: tl.constexpr, H: tl.constexpr,
    stride_w_v, stride_w_h,
    stride_wr_c, stride_wr_h,
    BLOCK_SIZE_V: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
):
    pid_v = tl.program_id(0)
    offs_v = pid_v * BLOCK_SIZE_V + tl.arange(0, BLOCK_SIZE_V)
    mask_v = offs_v < V

    # We want to find the argmax for a block of V tokens over all C clusters.
    # Initialize max_sims and best_clusters
    max_sims = tl.zeros((BLOCK_SIZE_V,), dtype=tl.float32) - float('inf')
    best_clusters = tl.zeros((BLOCK_SIZE_V,), dtype=tl.int32)

    # Accumulate dot products over the H dimension in blocks
    # Actually, we can load a block of w_reduced [BLOCK_C, H] and a block of w [BLOCK_V, H]
    # and compute matmul: [BLOCK_V, H] @ [H, BLOCK_C] -> [BLOCK_V, BLOCK_C]
    
    offs_h = tl.arange(0, BLOCK_SIZE_H)

    for c_start in range(0, C, BLOCK_SIZE_C):
        offs_c = c_start + tl.arange(0, BLOCK_SIZE_C)
        mask_c = offs_c < C
        
        # Accumulator for SIMS: [BLOCK_V, BLOCK_C]
        acc = tl.zeros((BLOCK_SIZE_V, BLOCK_SIZE_C), dtype=tl.float32)
        
        # We process H in blocks of BLOCK_SIZE_H. 
        # Since H is small (768), we usually use BLOCK_SIZE_H=128 or H itself
        for h_start in range(0, H, BLOCK_SIZE_H):
            curr_offs_h = h_start + offs_h
            mask_h = curr_offs_h < H
            
            # Load W block: [BLOCK_V, BLOCK_H]
            w_ptrs = w_ptr + (offs_v[:, None] * stride_w_v + curr_offs_h[None, :] * stride_w_h)
            w_mask = mask_v[:, None] & mask_h[None, :]
            w_block = tl.load(w_ptrs, mask=w_mask, other=0.0)
            
            # Load W_reduced block: [BLOCK_C, BLOCK_H]
            wr_ptrs = w_reduced_ptr + (offs_c[:, None] * stride_wr_c + curr_offs_h[None, :] * stride_wr_h)
            wr_mask = mask_c[:, None] & mask_h[None, :]
            wr_block = tl.load(wr_ptrs, mask=wr_mask, other=0.0)
            
            # Matmul: W_block @ (W_reduced_block.T) -> [BLOCK_V, BLOCK_C]
            # w_block is     [BLOCK_V, BLOCK_H]
            # wr_block is    [BLOCK_C, BLOCK_H]
            # triton matmul requires operands, we can transpose wr_block
            acc += tl.dot(w_block, tl.trans(wr_block))
            
        # Post-process SIMS for this C block
        # Mask out out-of-bounds C
        acc = tl.where(mask_c[None, :], acc, -float('inf'))
        
        # Find local max and argmax across C
        local_max = tl.max(acc, axis=1)
        local_argmax = tl.argmax(acc, axis=1)
        
        # Update global max and argmax
        update_mask = local_max > max_sims
        max_sims = tl.where(update_mask, local_max, max_sims)
        best_clusters = tl.where(update_mask, c_start + local_argmax, best_clusters)
        
    tl.store(new_cluster_ids_ptr + offs_v, best_clusters, mask=mask_v)


def fused_sim_argmax(w: torch.Tensor, w_reduced: torch.Tensor) -> torch.Tensor:
    V, H = w.shape
    C, H2 = w_reduced.shape
    assert H == H2

    new_cluster_ids = torch.empty(V, device=w.device, dtype=torch.long)

    # Tuning constants for max throughput
    BLOCK_SIZE_V = 64
    BLOCK_SIZE_C = 64
    BLOCK_SIZE_H = 64

    # Determine num_warps and num_stages based on block sizes
    num_warps = 4
    num_stages = 3
    if w.dtype == torch.bfloat16 or w.dtype == torch.float16:
        num_warps = 4
        num_stages = 4

    grid = lambda META: (
        triton.cdiv(V, META['BLOCK_SIZE_V']),
    )

    w = w.contiguous()
    w_reduced = w_reduced.contiguous()

    fused_sim_argmax_kernel[grid](
        w, w_reduced, new_cluster_ids,
        V, C, H,
        w.stride(0), w.stride(1),
        w_reduced.stride(0), w_reduced.stride(1),
        BLOCK_SIZE_V=BLOCK_SIZE_V,
        BLOCK_SIZE_C=BLOCK_SIZE_C,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        num_warps=num_warps,
        num_stages=num_stages
    )

    return new_cluster_ids
