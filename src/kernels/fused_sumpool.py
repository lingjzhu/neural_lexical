import torch
import triton
import triton.language as tl

# -----------------------------------------------------------------------------
# 1. Forward Kernel (Tuned)
# -----------------------------------------------------------------------------
@triton.jit
def feature_counting_fwd_kernel(
    X_ptr, W_ptr, Y_ptr, Mask_ptr,
    stride_xb, stride_xt, stride_xd,
    stride_wv, stride_wd,
    stride_yb, stride_yv,
    stride_mb, stride_mt, stride_mv,
    B, T, V, D,
    BLOCK_T: tl.constexpr, BLOCK_V: tl.constexpr, BLOCK_D: tl.constexpr
):
    # Grid: (B, V_chunk)
    pid_b = tl.program_id(0)
    pid_v = tl.program_id(1)

    # Base Pointers
    y_ptr_block = Y_ptr + pid_b * stride_yb + (pid_v * BLOCK_V + tl.arange(0, BLOCK_V)) * stride_yv
    w_ptr_base = W_ptr + (pid_v * BLOCK_V + tl.arange(0, BLOCK_V)[:, None]) * stride_wv
    x_ptr_base = X_ptr + pid_b * stride_xb

    # Accumulator (Always FP32 for precision)
    acc_sum = tl.zeros([BLOCK_V], dtype=tl.float32)
    
    # Pre-calc bit weights: [1, 2, 4, ..., 2^31]
    pack_weights = (1 << tl.arange(0, 32))[:, None]

    # Loop over Sequence T
    # BLOCK_T must be 32 for bit packing
    for t_idx in range(0, T, BLOCK_T):
        offs_t = t_idx + tl.arange(0, BLOCK_T)
        acc_dot = tl.zeros([BLOCK_T, BLOCK_V], dtype=tl.float32)

        # Inner Loop over Dimension D
        for d_idx in range(0, D, BLOCK_D):
            offs_d = d_idx + tl.arange(0, BLOCK_D)
            
            # Load X [32, 64]
            x_ptr = x_ptr_base + (offs_t[:, None] * stride_xt) + (offs_d[None, :] * stride_xd)
            x_mask = (offs_t[:, None] < T) & (offs_d[None, :] < D)
            x = tl.load(x_ptr, mask=x_mask, other=0.0)

            # Load W [128, 64]
            # Note: W is re-loaded for every T chunk. 
            # L2 cache (96MB on RTX 6000) handles this efficiently.
            w_ptr = w_ptr_base + (offs_d[None, :] * stride_wd)
            w_mask = (pid_v * BLOCK_V + tl.arange(0, BLOCK_V)[:, None] < V) & (offs_d[None, :] < D)
            w = tl.load(w_ptr, mask=w_mask, other=0.0)

            # Matmul: (32,64) @ (64,128) -> (32,128)
            # allow_tf32=False for high precision in FP32 mode
            acc_dot += tl.dot(x, w.T, allow_tf32=False)

        # Activation & Pooling
        relu_mask = acc_dot > 0.0
        relu_out = tl.where(relu_mask, acc_dot, 0.0)
        
        # Accumulate Sum over T
        acc_sum += tl.sum(relu_out, axis=0)

        # Bit Packing
        # Cast bool -> int32 -> multiply weight -> sum
        packed = tl.sum(relu_mask.to(tl.int32) * pack_weights, axis=0)

        # Store Mask
        mask_ptr_offset = (pid_b * stride_mb) + \
                          ((t_idx // 32) * stride_mt) + \
                          ((pid_v * BLOCK_V + tl.arange(0, BLOCK_V)) * stride_mv)
        mask_check = (pid_v * BLOCK_V + tl.arange(0, BLOCK_V)) < V
        tl.store(Mask_ptr + mask_ptr_offset, packed, mask=mask_check)

    # Store Result Y
    y_mask = (pid_v * BLOCK_V + tl.arange(0, BLOCK_V)) < V
    tl.store(y_ptr_block, acc_sum, mask=y_mask)

# -----------------------------------------------------------------------------
# 2. Backward Kernel dX
# -----------------------------------------------------------------------------
@triton.jit
def feature_counting_bwd_x_kernel(
    DX_ptr, W_ptr, GradY_ptr, Mask_ptr,
    stride_dxb, stride_dxt, stride_dxd,
    stride_wv, stride_wd,
    stride_gyb, stride_gyv,
    stride_mb, stride_mt, stride_mv,
    B, T, V, D,
    BLOCK_T: tl.constexpr, BLOCK_V: tl.constexpr, BLOCK_D: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1) 
    t_start = pid_t * BLOCK_T
    
    offs_t = t_start + tl.arange(0, BLOCK_T)
    unpack_shifter = tl.arange(0, 32)[:, None] 

    # Loop D to reconstruct full dX row
    for d_idx in range(0, D, BLOCK_D):
        offs_d = d_idx + tl.arange(0, BLOCK_D)
        acc_dx_part = tl.zeros([BLOCK_T, BLOCK_D], dtype=tl.float32)
        
        # Accumulate gradients from all Vocabulary words
        for v_idx in range(0, V, BLOCK_V):
            # Load Mask
            mask_offset = (pid_b * stride_mb) + (pid_t * stride_mt) + ((v_idx + tl.arange(0, BLOCK_V)) * stride_mv)
            mask_check = (v_idx + tl.arange(0, BLOCK_V)) < V
            packed_mask = tl.load(Mask_ptr + mask_offset, mask=mask_check, other=0)
            
            # Unpack Mask
            unpacked_mask = ((packed_mask[None, :] >> unpack_shifter) & 1).to(tl.int1)

            # Load GradY
            gy_ptr = GradY_ptr + (pid_b * stride_gyb) + ((v_idx + tl.arange(0, BLOCK_V)) * stride_gyv)
            gy = tl.load(gy_ptr, mask=mask_check, other=0.0)
            
            # Load W
            w_ptr = W_ptr + ((v_idx + tl.arange(0, BLOCK_V)[:, None]) * stride_wv) + (offs_d[None, :] * stride_wd)
            w_check = ((v_idx + tl.arange(0, BLOCK_V)[:, None]) < V) & (offs_d[None, :] < D)
            w = tl.load(w_ptr, mask=w_check, other=0.0)
            
            # Effective Gradient dZ = GradY * Mask
            # Cast to w.dtype (Handles BF16 correctly)
            dz = tl.where(unpacked_mask, gy[None, :], 0.0).to(w.dtype)
            
            # dX += dZ @ W
            acc_dx_part += tl.dot(dz, w, allow_tf32=False)
            
        # Store dX
        dx_ptr = DX_ptr + pid_b * stride_dxb + (offs_t[:, None] * stride_dxt) + (offs_d[None, :] * stride_dxd)
        dx_mask = (offs_t[:, None] < T) & (offs_d[None, :] < D)
        tl.store(dx_ptr, acc_dx_part, mask=dx_mask)

# -----------------------------------------------------------------------------
# 3. Backward Kernel dW
# -----------------------------------------------------------------------------
@triton.jit
def feature_counting_bwd_w_kernel(
    DW_ptr, X_ptr, GradY_ptr, Mask_ptr,
    stride_dwv, stride_dwd,
    stride_xb, stride_xt, stride_xd,
    stride_gyb, stride_gyv,
    stride_mb, stride_mt, stride_mv,
    B, T, V, D,
    BLOCK_T: tl.constexpr, BLOCK_V: tl.constexpr, BLOCK_D: tl.constexpr
):
    pid_v = tl.program_id(0)
    pid_d = tl.program_id(1)
    
    offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    unpack_shifter = tl.arange(0, 32)[:, None] 
    
    acc_dw = tl.zeros([BLOCK_V, BLOCK_D], dtype=tl.float32)

    # Accumulate over Batch and Sequence
    for b_idx in range(B):
        for t_idx in range(0, T, BLOCK_T):
            offs_t = t_idx + tl.arange(0, BLOCK_T)
            
            # Load Mask
            mask_offset = (b_idx * stride_mb) + ((t_idx // 32) * stride_mt) + (offs_v * stride_mv)
            mask_check = offs_v < V
            packed_mask = tl.load(Mask_ptr + mask_offset, mask=mask_check, other=0)
            unpacked_mask = ((packed_mask[None, :] >> unpack_shifter) & 1).to(tl.int1)
            
            # Load GradY
            gy_ptr = GradY_ptr + (b_idx * stride_gyb) + (offs_v * stride_gyv)
            gy = tl.load(gy_ptr, mask=mask_check, other=0.0)
            
            # Calculate dZ
            dz = tl.where(unpacked_mask, gy[None, :], 0.0).to(X_ptr.dtype.element_ty)
            
            # Load X
            x_ptr = X_ptr + (b_idx * stride_xb) + (offs_t[:, None] * stride_xt) + (offs_d[None, :] * stride_xd)
            x_check = (offs_t[:, None] < T) & (offs_d[None, :] < D)
            x = tl.load(x_ptr, mask=x_check, other=0.0)
            
            # dW += dZ.T @ X
            acc_dw += tl.dot(dz.T, x, allow_tf32=False)

    # Store dW
    dw_ptr = DW_ptr + (offs_v[:, None] * stride_dwv) + (offs_d[None, :] * stride_dwd)
    dw_check = (offs_v[:, None] < V) & (offs_d[None, :] < D)
    tl.store(dw_ptr, acc_dw, mask=dw_check)


class FeatureCountFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w):
        B, T, D = x.shape
        V, _ = w.shape
        
        # 1. Pad T to multiple of 32
        pad_t = 0
        if T % 32 != 0:
            pad_t = 32 - (T % 32)
            # Pad temporal dim with 0
            x = torch.nn.functional.pad(x, (0, 0, 0, pad_t))
            T = x.shape[1]
        
        # 2. Output alloc
        y = torch.empty((B, V), device=x.device, dtype=x.dtype) # Match input dtype
        mask_packed = torch.empty((B, T // 32, V), device=x.device, dtype=torch.int32)

        # 3. Kernel Config
        grid = (B, triton.cdiv(V, 128))
        # num_warps=4, num_stages=3 helps compiler pipeline loads
        feature_counting_fwd_kernel[grid](
            x, w, y, mask_packed,
            x.stride(0), x.stride(1), x.stride(2),
            w.stride(0), w.stride(1),
            y.stride(0), y.stride(1),
            mask_packed.stride(0), mask_packed.stride(1), mask_packed.stride(2),
            B, T, V, D,
            BLOCK_T=32, BLOCK_V=128, BLOCK_D=64,
            num_warps=4, num_stages=3 
        )
        
        ctx.save_for_backward(x, w, mask_packed)
        ctx.pad_t = pad_t
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, w, mask_packed = ctx.saved_tensors
        pad_t = ctx.pad_t
        B, T, D = x.shape
        V, _ = w.shape
        
        grad_output = grad_output.contiguous()
        dx = torch.empty_like(x)
        dw = torch.empty_like(w)
        
        # dX Kernel
        grid_dx = (B, T // 32)
        feature_counting_bwd_x_kernel[grid_dx](
            dx, w, grad_output, mask_packed,
            dx.stride(0), dx.stride(1), dx.stride(2),
            w.stride(0), w.stride(1),
            grad_output.stride(0), grad_output.stride(1),
            mask_packed.stride(0), mask_packed.stride(1), mask_packed.stride(2),
            B, T, V, D,
            BLOCK_T=32, BLOCK_V=128, BLOCK_D=64,
            num_warps=4, num_stages=3
        )
        
        # dW Kernel
        grid_dw = (triton.cdiv(V, 128), triton.cdiv(D, 64))
        feature_counting_bwd_w_kernel[grid_dw](
            dw, x, grad_output, mask_packed,
            dw.stride(0), dw.stride(1),
            x.stride(0), x.stride(1), x.stride(2),
            grad_output.stride(0), grad_output.stride(1),
            mask_packed.stride(0), mask_packed.stride(1), mask_packed.stride(2),
            B, T, V, D,
            BLOCK_T=32, BLOCK_V=128, BLOCK_D=64,
            num_warps=4, num_stages=3
        )
        
        if pad_t > 0:
            dx = dx[:, :-pad_t, :]
            
        return dx, dw

def feature_sum_pool(x, w):
    return FeatureCountFunction.apply(x, w)