import copy
from collections.abc import Callable
from typing import Optional, Union, List

import torch
import torch.nn as nn

from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig

# Standalone configuration classes
class T5Gemma2TextConfig(PretrainedConfig):
    model_type = "t5gemma2_text"
    def __init__(
        self,
        vocab_size=256000,
        hidden_size=3072,
        intermediate_size=24576,
        num_hidden_layers=28,
        num_attention_heads=16,
        num_key_value_heads=16,
        head_dim=256,
        hidden_activation="gelu_pytorch_tanh",
        dropout_rate=0.0,
        attention_dropout=0.0,
        max_position_embeddings=8192,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=2,
        eos_token_id=1,
        sliding_window=4096,
        layer_types=None,
        query_pre_attn_scalar=None,
        attn_logit_softcapping=None,
        final_logit_softcapping=None,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_activation = hidden_activation
        self.dropout_rate = dropout_rate
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.sliding_window = sliding_window
        self.layer_types = layer_types or ["full_attention"] * num_hidden_layers
        self.query_pre_attn_scalar = query_pre_attn_scalar or (self.head_dim**0.5)
        self.attn_logit_softcapping = attn_logit_softcapping
        self.final_logit_softcapping = final_logit_softcapping

class T5Gemma2Config(PretrainedConfig):
    model_type = "t5gemma2"
    def __init__(self, text_config=None, eoi_token_index=256000, **kwargs):
        super().__init__(**kwargs)
        if isinstance(text_config, dict):
            text_config = T5Gemma2TextConfig(**text_config)
        self.text_config = text_config or T5Gemma2TextConfig()
        self.eoi_token_index = eoi_token_index

# Standalone RMSNorm
class T5Gemma2RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))
    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    def forward(self, x):
        output = self._norm(x.float())
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)

# Standalone MLP
class T5Gemma2MLP(nn.Module):
    def __init__(self, config: T5Gemma2TextConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_activation]
        self.dropout = nn.Dropout(config.dropout_rate)
    def forward(self, x):
        hidden_states = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        hidden_states = self.dropout(hidden_states)
        return self.down_proj(hidden_states)

# Rotary Embedding implementation
def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class T5Gemma2RotaryEmbedding(nn.Module):
    def __init__(self, config: T5Gemma2TextConfig, device=None):
        super().__init__()
        self.head_dim = config.head_dim
        self.dim = self.head_dim
        self.base = 10000
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x, position_ids, layer_type=None):
        inv_freq_expanded = self.inv_freq[None, None, :].expand(position_ids.shape[0], -1, -1)
        position_ids_expanded = position_ids[:, :, None].float()
        freqs = (inv_freq_expanded * position_ids_expanded).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()

# Simple Attention
def eager_attention_forward(module, query, key, value, attention_mask, dropout=0.0, scaling=1.0, softcap=None):
    if softcap is None and attention_mask is not None and not torch.is_grad_enabled():
        # Try to use SDPA for inference if no softcap
        try:
            return torch.nn.functional.scaled_dot_product_attention(
                query * scaling, key, value, attn_mask=attention_mask, dropout_p=dropout if module.training else 0.0
            ), None
        except Exception:
            pass

    query = query * scaling
    if softcap is not None:
        query = query / softcap
        query = torch.tanh(query)
        query = query * softcap
    
    attn_weights = torch.matmul(query, key.transpose(2, 3))
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    if dropout > 0.0:
        attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value)
    return attn_output, attn_weights

class T5Gemma2SelfAttention(nn.Module):
    def __init__(self, config: T5Gemma2TextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.scaling = config.query_pre_attn_scalar**-0.5
        self.attention_dropout = config.attention_dropout
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)
        self.q_norm = T5Gemma2RMSNorm(dim=self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = T5Gemma2RMSNorm(dim=self.head_dim, eps=config.rms_norm_eps)
        self.attn_logit_softcapping = config.attn_logit_softcapping

    def forward(self, hidden_states, position_embeddings=None, attention_mask=None, **kwargs):
        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)
        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        # Manually handle GQA if needed
        if self.num_key_value_heads != self.num_heads:
            num_groups = self.num_heads // self.num_key_value_heads
            key_states = key_states.repeat_interleave(num_groups, dim=1)
            value_states = value_states.repeat_interleave(num_groups, dim=1)

        # Optimization: Use SDPA if possible
        if self.attn_logit_softcapping is None:
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states, key_states, value_states, 
                attn_mask=attention_mask, 
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=False,
                scale=self.scaling
            )
            attn_weights = None
        else:
            attn_output, attn_weights = eager_attention_forward(
                self, query_states, key_states, value_states, attention_mask,
                dropout=self.attention_dropout if self.training else 0.0,
                scaling=self.scaling, softcap=self.attn_logit_softcapping
            )
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

class T5Gemma2EncoderLayer(nn.Module):
    def __init__(self, config: T5Gemma2TextConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = T5Gemma2SelfAttention(config, layer_idx)
        self.pre_self_attn_layernorm = T5Gemma2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_self_attn_layernorm = T5Gemma2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = T5Gemma2MLP(config)
        self.pre_feedforward_layernorm = T5Gemma2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = T5Gemma2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states, position_embeddings=None, attention_mask=None, **kwargs):
        residual = hidden_states
        hidden_states = self.pre_self_attn_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(hidden_states, position_embeddings, attention_mask, **kwargs)
        hidden_states = residual + self.dropout(self.post_self_attn_layernorm(hidden_states))
        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + self.dropout(self.post_feedforward_layernorm(hidden_states))
        return hidden_states

class T5Gemma2TextScaledWordEmbedding(nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim, padding_idx, embed_scale=1.0, eoi_token_index=256000):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.register_buffer("embed_scale", torch.tensor(embed_scale), persistent=False)
        self.eoi_token_index = eoi_token_index
        self.eoi_embedding = nn.Parameter(torch.zeros(embedding_dim))
    def forward(self, input_ids):
        input_embeddings = super().forward(input_ids) * self.embed_scale.to(self.weight.dtype)
        mask = (input_ids == self.eoi_token_index)
        if mask.any():
            input_embeddings[mask] = self.eoi_embedding.to(input_embeddings.dtype)
        return input_embeddings

class T5Gemma2PreTrainedModel(PreTrainedModel):
    config_class = T5Gemma2TextConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None: module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None: module.weight.data[module.padding_idx].zero_()

class T5Gemma2TextEncoder(T5Gemma2PreTrainedModel):
    def __init__(self, config: T5Gemma2TextConfig, eoi_token_index: int = 256000):
        super().__init__(config)
        self.embed_tokens = T5Gemma2TextScaledWordEmbedding(
            config.vocab_size, config.hidden_size, config.pad_token_id, 
            embed_scale=config.hidden_size**0.5, eoi_token_index=eoi_token_index
        )
        self.layers = nn.ModuleList([T5Gemma2EncoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = T5Gemma2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = T5Gemma2RotaryEmbedding(config)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self): return self.embed_tokens
    def set_input_embeddings(self, value): self.embed_tokens = value

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, inputs_embeds=None, **kwargs):
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if position_ids is None:
            position_ids = torch.arange(0, inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)
        
        if attention_mask is not None and attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
            extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(inputs_embeds.dtype).min
        else:
            extended_attention_mask = None

        hidden_states = self.dropout(inputs_embeds)
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    layer, 
                    hidden_states, 
                    (cos, sin), 
                    extended_attention_mask,
                    use_reentrant=False
                )
            else:
                hidden_states = layer(hidden_states, position_embeddings=(cos, sin), attention_mask=extended_attention_mask)
        
        hidden_states = self.norm(hidden_states)
        return BaseModelOutput(last_hidden_state=hidden_states)

def patch_t5gemma2(model):
    """
    Patch T5Gemma2TextEncoder instances with Liger kernels for RMSNorm and MLP.
    """
    try:
        from liger_kernel.transformers.monkey_patch import _bind_method_to_module, _patch_rms_norm_module
        from liger_kernel.transformers.geglu import LigerGEGLUMLP
    except ImportError:
        print("liger_kernel not found. Skipping patching.")
        return

    def patch_rms(module):
        if isinstance(module, T5Gemma2RMSNorm):
            # Gemma 2/3 settings: offset=1.0, casting_mode="gemma", in_place=False
            _patch_rms_norm_module(module, offset=1.0, casting_mode="gemma", in_place=False)
            # Ensure it has the correct name for identification
            _bind_method_to_module(module, "_get_name", lambda self: "LigerRMSNormForGemma")

    def patch_mlp(module):
        if isinstance(module, T5Gemma2MLP):
            _bind_method_to_module(module, "forward", LigerGEGLUMLP.forward)
            _bind_method_to_module(module, "_get_name", lambda self: "LigerGEGLUMLP")

    for name, module in model.named_modules():
        patch_rms(module)
        patch_mlp(module)

__all__ = ["T5Gemma2TextEncoder", "T5Gemma2Config", "T5Gemma2TextConfig", "T5Gemma2PreTrainedModel", "patch_t5gemma2"]