import torch
import math
from torch.nn.functional import scaled_dot_product_attention

from unsloth.kernels import fast_linear_forward
from unsloth.models.llama import (
    fast_rms_layernorm_inference, 
    LlamaRotaryEmbedding,
)
from unsloth.models.loader import FastModel
from peft import get_peft_model

import unsloth.kernels.utils as utils
import unsloth.kernels.fast_lora as fast_lora
_original_fast_dequantize = utils.fast_dequantize

def _patched_fast_dequantize(W, quant_state=None, out=None, use_global_buffer=False):
    if W is not None:
        if getattr(W, "dtype", None) == torch.float8_e4m3fn or (hasattr(W, "data") and W.data.dtype == torch.float8_e4m3fn):
            from fast_fp8 import weight_dequant
            return weight_dequant(W, quant_state)
        # Handle the case where Hugging Face's trainer or accelerate forcefully upcast the FP8 Parameter into a standard floating point type before forward pass
        if isinstance(quant_state, torch.Tensor) and quant_state.numel() == 1 and str(getattr(W, "dtype", "")).startswith("torch."):
            return W * quant_state.to(W.dtype)
            
    return _original_fast_dequantize(W, quant_state, out, use_global_buffer)

utils.fast_dequantize = _patched_fast_dequantize
fast_lora.fast_dequantize = _patched_fast_dequantize

def get_lora_qkv(q_proj, k_proj, v_proj, x):
    from unsloth.kernels.utils import get_lora_parameters, _maybe_fake_quantize_activations
    from unsloth.kernels.fast_lora import LoRA_QKV
    x_q = _maybe_fake_quantize_activations(x, q_proj)
    QW, QW_quant, QA, QB, QS = get_lora_parameters(q_proj)
    KW, KW_quant, KA, KB, KS = get_lora_parameters(k_proj)
    VW, VW_quant, VA, VB, VS = get_lora_parameters(v_proj)
    return LoRA_QKV.apply(
        x_q,
        QW, QW_quant, QA, QB, QS,
        KW, KW_quant, KA, KB, KS,
        VW, VW_quant, VA, VB, VS,
        True
    )

def get_lora_mlp(gate_proj, up_proj, down_proj, x):
    from unsloth.kernels.utils import get_lora_parameters, _maybe_fake_quantize_activations
    from unsloth.kernels.fast_lora import LoRA_MLP
    from unsloth.kernels.swiglu import swiglu_fg_kernel, swiglu_DWf_DW_dfg_kernel
    x_g = _maybe_fake_quantize_activations(x, gate_proj)
    gateW, gateW_quant, gateA, gateB, gateS = get_lora_parameters(gate_proj)
    upW, upW_quant, upA, upB, upS = get_lora_parameters(up_proj)
    downW, downW_quant, downA, downB, downS = get_lora_parameters(down_proj)
    return LoRA_MLP.apply(
        x_g,
        gateW, gateW_quant, gateA, gateB, gateS,
        upW, upW_quant, upA, upB, upS,
        downW, downW_quant, downA, downB, downS,
        swiglu_fg_kernel, swiglu_DWf_DW_dfg_kernel, True
    )

def get_lora_w(linear, x):
    from unsloth.kernels.utils import get_lora_parameters, _maybe_fake_quantize_activations
    from unsloth.kernels.fast_lora import LoRA_W
    x_w = _maybe_fake_quantize_activations(x, linear)
    W, W_quant, A, B, S = get_lora_parameters(linear)
    return LoRA_W.apply(x_w, W, W_quant, A, B, S)

def LLaDABlock_fast_attention(
    self,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_bias = None,
    layer_past = None,
    use_cache: bool = False,
):
    B, T, C = q.size()
    dtype = k.dtype

    if getattr(self, "q_norm", None) is not None and getattr(self, "k_norm", None) is not None:
        q = self.q_norm(q).to(dtype=dtype)
        k = self.k_norm(k).to(dtype=dtype)

    q = q.view(B, T, self.config.n_heads, C // self.config.n_heads).transpose(1, 2)
    k = k.view(B, T, self.config.effective_n_kv_heads, C // self.config.n_heads).transpose(1, 2)
    v = v.view(B, T, self.config.effective_n_kv_heads, C // self.config.n_heads).transpose(1, 2)

    if layer_past is not None:
        past_key, past_value = layer_past
        k = torch.cat((past_key, k), dim=-2)
        v = torch.cat((past_value, v), dim=-2)

    present = (k, v) if use_cache else None
    query_len, key_len = q.shape[-2], k.shape[-2]

    if self.config.rope:
        q, k = self.rotary_emb(q, k)

    if attention_bias is not None:
        attention_bias = self._cast_attn_bias(
            attention_bias[:, :, key_len - query_len : key_len, :key_len], dtype
        )

    att = self._scaled_dot_product_attention(
        q, k, v,
        attn_mask=attention_bias,
        dropout_p=0.0 if not self.training else self.config.attention_dropout,
        is_causal=False,
    )
    att = att.transpose(1, 2).contiguous().view(B, T, C)

    if getattr(self.attn_out, "lora_A", None) is not None:
        out = get_lora_w(self.attn_out, att)
    else:
        out = fast_linear_forward(self.attn_out, att)
        
    return out, present

def LLaDALlamaBlock_fast_forward(
    self,
    x: torch.Tensor,
    attention_bias = None,
    layer_past = None,
    use_cache: bool = False,
):
    og_x = x
    x = self.attn_norm(x)

    # Q, K, V projections
    if getattr(self.q_proj, "lora_A", None) is not None:
        q, k, v = get_lora_qkv(self.q_proj, self.k_proj, self.v_proj, x)
    else:
        q = fast_linear_forward(self.q_proj, x)
        k = fast_linear_forward(self.k_proj, x)
        v = fast_linear_forward(self.v_proj, x)

    if getattr(self, "_activation_checkpoint_fn", None) is not None:
        att, cache = self._activation_checkpoint_fn(
            self.attention, q, k, v, attention_bias, layer_past=layer_past, use_cache=use_cache
        )
    else:
        att, cache = self.attention(q, k, v, attention_bias, layer_past=layer_past, use_cache=use_cache)

    x = og_x + self.dropout(att)

    og_x = x
    if getattr(self, "_activation_checkpoint_fn", None) is not None:
        x = self._activation_checkpoint_fn(self.ff_norm, x)
    else:
        x = self.ff_norm(x)
        
    # MLP projections
    if getattr(self.ff_proj, "lora_A", None) is not None:
        x = get_lora_mlp(self.ff_proj, self.up_proj, self.ff_out, x)
    else:
        x_ff = fast_linear_forward(self.ff_proj, x)
        x_up = fast_linear_forward(self.up_proj, x)
        
        if getattr(self, "_activation_checkpoint_fn", None) is not None:
            x_ff = self._activation_checkpoint_fn(self.act, x_ff)
        else:
            x_ff = self.act(x_ff)
            
        x = x_ff * x_up
        x = fast_linear_forward(self.ff_out, x)
        
    x = self.dropout(x)
    x = og_x + x

    return x, cache


class FastLLaDAModel(FastModel):
    @staticmethod
    def from_pretrained(*args, **kwargs):
        from transformers import AutoModel
        kwargs["auto_model"] = AutoModel
        if "use_gradient_checkpointing" not in kwargs:
            kwargs["use_gradient_checkpointing"] = False
        kwargs.pop("load_in_fp8", None)  # Unsloth only supports FP8 for inference
        
        model, tokenizer = FastModel.from_pretrained(*args, **kwargs)
        
        # Patch dynamically loaded classes
        LLaDALlamaBlock = model.model.transformer.blocks[0].__class__
        LLaDABlock = LLaDALlamaBlock.__bases__[0]
        LLaDABlock.attention = LLaDABlock_fast_attention
        LLaDALlamaBlock.forward = LLaDALlamaBlock_fast_forward
        
        return model, tokenizer

    @staticmethod
    def get_peft_model(
        model,
        r = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "attn_out", "ff_proj", "up_proj", "ff_out"],
        lora_alpha = 16,
        lora_dropout = 0,
        bias = "none",
        layers_to_transform = None,
        layers_pattern = None,
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
        max_seq_length = 2048,
        use_rslora = False,
        init_lora_weights = True,
        loftq_config = None,
        temporary_location = "_unsloth_temporary_saved_buffers",
        **kwargs,
    ):
        # We fall back to Unsloth's llama peft model internally so we get fast kernels
        from unsloth.models.llama import FastLlamaModel
        return FastLlamaModel.get_peft_model(
            model=model,
            r=r,
            target_modules=target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=bias,
            layers_to_transform=layers_to_transform,
            layers_pattern=layers_pattern,
            use_gradient_checkpointing=use_gradient_checkpointing,
            random_state=random_state,
            max_seq_length=max_seq_length,
            use_rslora=use_rslora,
            init_lora_weights=init_lora_weights,
            loftq_config=loftq_config,
            temporary_location=temporary_location,
            **kwargs
        )
