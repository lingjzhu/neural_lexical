import os
import torch
import unsloth.kernels.utils as utils

# Save original
original_fast_dequantize = utils.fast_dequantize

def patched_fast_dequantize(W, quant_state=None, out=None, use_global_buffer=False):
    if W is not None and getattr(W, "dtype", None) == torch.float8_e4m3fn:
        from fast_fp8 import weight_dequant
        return weight_dequant(W, quant_state)
    return original_fast_dequantize(W, quant_state, out, use_global_buffer)

# Monkey patch unsloth
utils.fast_dequantize = patched_fast_dequantize
print("Patched fast_dequantize!")
