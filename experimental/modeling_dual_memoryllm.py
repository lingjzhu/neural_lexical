import math
from typing import Optional, List, Tuple
from collections.abc import Callable

import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutput

from modeling_modernbert import (
    ModernBertConfig,
    ModernBertModel,
    ModernBertPreTrainedModel,
    ModernBertEmbeddings,
    ModernBertMLP,
    ModernBertRotaryEmbedding,
    ModernBertAttention,
    ModernBertPredictionHead,
    create_bidirectional_mask,
    create_bidirectional_sliding_window_mask,
)
from transformers.utils import add_start_docstrings, add_start_docstrings_to_model_forward
from typing import Any

TransformersKwargs = Any
Unpack = Any
check_model_inputs = lambda x: x

# Import Engram functionality from the local module
from engram import Engram, NgramHashMapping, EngramConfig, MultiHeadEmbedding, ShortConv

import torch.nn.init as init


class DualMemoryLLMConfig(ModernBertConfig):
    """
    Configuration for DualMemoryLLM extending ModernBERT Config.
    """
    def __init__(self, 
                 engram_layer_ids=None,
                 engram_vocab_size=None,
                 max_ngram_size=3,
                 n_embed_per_ngram=512,
                 n_head_per_ngram=8,
                 tokenizer_name_or_path="answerdotai/ModernBERT-base",
                 engram_pad_id=2, # separate from bert pad_token_id
                 engram_seed=0,
                 engram_kernel_size=4,
                 engram_hc_mult=1, # Default backbone config for engram mock is 4, but ModernBERT doesn't use hyperconnections. We will use 1 for simplicity or adapt if needed.
                 **kwargs):
        
        super().__init__(**kwargs)
        if not hasattr(self, "layer_types") or self.layer_types is None:
            self.layer_types = ["full_attention"] * getattr(self, "num_hidden_layers", 4)
        if not hasattr(self, "rope_parameters") or self.rope_parameters is None:
            self.rope_parameters = {
                "full_attention": {"rope_type": "default", "rope_theta": 10000.0},
                "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
            }
        
        # Engram Specific parameters
        self.engram_layer_ids = engram_layer_ids if engram_layer_ids is not None else [1, 15]
        self.engram_vocab_size = engram_vocab_size if engram_vocab_size is not None else [129280*5, 129280*5]
        self.max_ngram_size = max_ngram_size
        self.n_embed_per_ngram = n_embed_per_ngram
        self.n_head_per_ngram = n_head_per_ngram
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.engram_pad_id = engram_pad_id
        self.engram_seed = engram_seed
        self.engram_kernel_size = engram_kernel_size
        self.engram_hc_mult = engram_hc_mult
        self.use_ngram_memory = kwargs.pop("use_ngram_memory", True)


class DualMemoryLLMEngram(nn.Module):
    """
    Adapted Engram module to fit within DualMemoryLLM.
    Re-uses NgramHashMapping logic.
    """
    def __init__(self, config: DualMemoryLLMConfig, layer_id: int, hash_mapping: NgramHashMapping):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.hash_mapping = hash_mapping
        
        engram_vocab_sizes_for_layer = [x for y in self.hash_mapping.vocab_size_across_layers[self.layer_id] for x in y]
        
        self.multi_head_embedding = MultiHeadEmbedding(
            list_of_N = engram_vocab_sizes_for_layer,
            D = config.n_embed_per_ngram // config.n_head_per_ngram,
        )
        
        self.short_conv = ShortConv(
            hidden_size = config.hidden_size,
            kernel_size = config.engram_kernel_size,
            dilation    = config.max_ngram_size,
            hc_mult     = config.engram_hc_mult, 
        )
        
        engram_hidden_size = (config.max_ngram_size - 1) * config.n_embed_per_ngram
        
        self.value_proj = nn.Linear(engram_hidden_size, config.hidden_size)
        
        # Since ModernBert doesn't have hc_mult natively in its self-attention (it's typical Transformer), 
        # we will set hc_mult=1 for Engram integration unless we expand ModernBERT states to [B, L, HC_MULT, D].
        # For simplicity and standard Transformer compat, we treat the hidden state as [B, L, D] -> [B, L, 1, D] for Engram.
        self.key_projs = nn.ModuleList(
            [nn.Linear(engram_hidden_size, config.hidden_size) for _ in range(config.engram_hc_mult)]
        )
        self.norm1 = nn.ModuleList([nn.RMSNorm(config.hidden_size) for _ in range(config.engram_hc_mult)])
        self.norm2 = nn.ModuleList([nn.RMSNorm(config.hidden_size) for _ in range(config.engram_hc_mult)])

    def forward(self, hidden_states, input_ids):
        """
        hidden_states: [B, L, D]
        input_ids: [B, L]
        """
        B, L, D = hidden_states.shape
        # Add the hc_mult dimension -> [B, L, HC_MULT, D]
        hidden_states_hc = hidden_states.unsqueeze(2).expand(-1, -1, self.config.engram_hc_mult, -1)
        
        # Fetch hashed IDs from cpu/numpy Mapping and move to device
        # Handle input_ids being potentially a numpy array or a torch tensor
        if isinstance(input_ids, torch.Tensor):
            input_ids_np = input_ids.cpu().numpy()
        else:
            input_ids_np = input_ids

        hash_input_ids = torch.from_numpy(self.hash_mapping.hash(input_ids_np)[self.layer_id])
        hash_input_ids = hash_input_ids.to(hidden_states.device)
        
        # Lookup embeddings
        embeddings = self.multi_head_embedding(hash_input_ids).flatten(start_dim=-2)
        
        gates = []
        for hc_idx in range(self.config.engram_hc_mult):
            key = self.key_projs[hc_idx](embeddings)
            normed_key = self.norm1[hc_idx](key)
            query = hidden_states_hc[:,:,hc_idx,:]
            normed_query = self.norm2[hc_idx](query)
            
            gate = (normed_key * normed_query).sum(dim=-1) / math.sqrt(self.config.hidden_size)
            gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
            gate = gate.sigmoid().unsqueeze(-1)
            gates.append(gate)
            
        gates = torch.stack(gates, dim=2)
        value = gates * self.value_proj(embeddings).unsqueeze(2)
        output = value + self.short_conv(value)
        
        # Reduce back to [B, L, D] assuming hc_mult is 1, or average/sum over hc_mult
        output = output.mean(dim=2) 
        
        return output

class DualMemoryLLMEncoderLayer(nn.Module):
    def __init__(self, config: DualMemoryLLMConfig, layer_idx: int, hash_mapping: NgramHashMapping = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        if layer_idx == 0:
            self.attn_norm = nn.Identity()
        else:
            self.attn_norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)
            
        # Pathway 1: Contextual Reasoning (Self-Attention)
        self.attn = ModernBertAttention(config=config, layer_idx=layer_idx)
        
        # Pathway 3: Token-Level Memory FFN (Uses original token embeddings)
        self.token_memory_norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)
        self.token_memory_mlp = ModernBertMLP(config)
        
        # Layer MLP (Used after standard Attention in Transformer, if we keep the architecture standard)
        # However, the dual-memory spec just says "Fusion Step: Sum the outputs of all three pathways"
        # We will keep the default MLP for the Contextual pathway output, then sum.
        self.mlp_norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)
        self.mlp = ModernBertMLP(config)
        
        self.attention_type = config.layer_types[layer_idx]
        
        # Pathway 2: Phrase-Level Memory (Engram)
        self.engram = None
        if config.use_ngram_memory and layer_idx in config.engram_layer_ids and hash_mapping is not None:
            self.engram = DualMemoryLLMEngram(config, layer_idx, hash_mapping)

    def forward(
        self,
        hidden_states: torch.Tensor,
        initial_token_embeddings: torch.Tensor,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        position_embeddings: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        
        # --- Pathway 1: Contextual Reasoning (Self-Attention) ---
        attn_output, _ = self.attn(
            self.attn_norm(hidden_states),
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            **kwargs,
        )
        # Apply standard FFN to the contextual output 
        contextual_out = hidden_states + attn_output
        contextual_out = contextual_out + self.mlp(self.mlp_norm(contextual_out))
        
        # --- Pathway 2: Phrase-Level Memory (Engram) ---
        phrase_memory_out = 0
        if self.engram is not None:
            phrase_memory_out = self.engram(hidden_states=hidden_states, input_ids=input_ids)
            
        # --- Pathway 3: Token-Level Memory ---
        # "Input: The initial, static token embeddings from the very first layer"
        # "Operation: Pass through a standard Feed-Forward Network"
        token_memory_out = self.token_memory_mlp(self.token_memory_norm(initial_token_embeddings))
        
        # --- Fusion Step ---
        hidden_states = contextual_out + phrase_memory_out + token_memory_out
        
        return hidden_states


class DualMemoryLLMModel(ModernBertPreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"layers\.\d+\.token_memory_mlp\..+",
        r"layers\.\d+\.engram\..+",
    ]
    def __init__(self, config: DualMemoryLLMConfig):
        super().__init__(config)
        self.config = config
        self.embeddings = ModernBertEmbeddings(config)
        
        # Initialize the global Engram Hash Mapping once for the entire model
        self.hash_mapping = None
        if self.config.use_ngram_memory:
            self.hash_mapping = NgramHashMapping(
                engram_vocab_size=config.engram_vocab_size,
                max_ngram_size=config.max_ngram_size,
                n_embed_per_ngram=config.n_embed_per_ngram,
                n_head_per_ngram=config.n_head_per_ngram,
                layer_ids=config.engram_layer_ids,
                tokenizer_name_or_path=config.tokenizer_name_or_path,
                pad_id=config.engram_pad_id,
                seed=config.engram_seed,
            )
        
        self.layers = nn.ModuleList(
            [DualMemoryLLMEncoderLayer(config, layer_idx, self.hash_mapping) for layer_idx in range(config.num_hidden_layers)]
        )
        self.final_norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)
        self.rotary_emb = ModernBertRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.tok_embeddings
    
    @torch.no_grad()
    def _init_weights(self, module):
        """
        Mirror the initialization logic in the top-level class.
        """
        # We can just delegate to a shared helper or just use the same logic
        std = getattr(self.config, "initializer_range", 0.02)
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.Conv1d):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, (nn.LayerNorm, nn.RMSNorm)):
            module.weight.data.fill_(1.0)
            if hasattr(module, "bias") and module.bias is not None:
                module.bias.data.zero_()
        
        try:
            super()._init_weights(module)
        except Exception:
            pass

    def set_input_embeddings(self, value):
        self.embeddings.tok_embeddings = value

    @check_model_inputs
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> BaseModelOutput:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        seq_len = inputs_embeds.shape[1] if inputs_embeds is not None else input_ids.shape[1]
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if position_ids is None:
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

        # Get the initial token embeddings from the embedding layer.
        initial_token_embeddings = self.embeddings(input_ids=input_ids, inputs_embeds=inputs_embeds)
        hidden_states = initial_token_embeddings
        
        # We need input_ids for Engram hashing. If not provided (i.e inputs_embeds used), we can't use Engram easily.
        # Fallback to dummy input_ids 
        if input_ids is None and self.config.use_ngram_memory:
             raise ValueError("DualMemoryLLM requires input_ids for Engram hashing pathway.")

        if not isinstance(attention_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "input_embeds": hidden_states,
                "attention_mask": attention_mask,
            }
            attention_mask_mapping = {
                "full_attention": create_bidirectional_mask(**mask_kwargs),
                "sliding_attention": create_bidirectional_sliding_window_mask(**mask_kwargs),
            }

        position_embeddings = {}
        for layer_type in self.config.layer_types:
            position_embeddings[layer_type] = self.rotary_emb(hidden_states, position_ids, layer_type=layer_type)


        for encoder_layer in self.layers:
            hidden_states = encoder_layer(
                hidden_states=hidden_states,
                initial_token_embeddings=initial_token_embeddings,  # PASSING TO PATHWAY 3
                input_ids=input_ids.cpu().numpy() if input_ids is not None else None, # PASSING TO PATHWAY 2
                attention_mask=attention_mask_mapping[encoder_layer.attention_type],
                position_embeddings=position_embeddings[encoder_layer.attention_type],
                **kwargs,
            )

        hidden_states = self.final_norm(hidden_states)

        return BaseModelOutput(last_hidden_state=hidden_states)


class AuthorshipEngramHead(nn.Module):
    def __init__(self, hidden_size, engram_hidden_size, k_selection=128):
        super().__init__()
        self.k = k_selection
        self.hidden_size = hidden_size
        
        # DeepSeek's Q-K Gating Projections
        self.value_proj = nn.Linear(engram_hidden_size, hidden_size)
        self.key_proj = nn.Linear(engram_hidden_size, hidden_size)
        
        # RMSNorms for stable dot-products
        self.norm_key = nn.RMSNorm(hidden_size)
        self.norm_query = nn.RMSNorm(hidden_size)

    def forward(self, hidden_states, engram_embeddings, attention_mask=None):
        """
        hidden_states: [B, L, D] - The dynamic contextual sequence
        engram_embeddings: [B, L, Engram_D] - The static fetched N-grams
        attention_mask: [B, L] - Padding mask
        """
        B, L, _ = hidden_states.shape
        
        # 1. Prepare Keys (Static) and Queries (Dynamic)
        key = self.key_proj(engram_embeddings)
        normed_key = self.norm_key(key)
        
        normed_query = self.norm_query(hidden_states)
        
        # 2. Compute Official DeepSeek Gate Scores
        raw_gate = (normed_key * normed_query).sum(dim=-1) / math.sqrt(self.hidden_size)
        
        # The signed square-root stabilization trick
        stable_gate = raw_gate.abs().clamp_min(1e-6).sqrt() * raw_gate.sign()
        
        # Final gate evaluated between 0 and 1
        gate_scores = stable_gate.sigmoid() 
        
        if attention_mask is not None:
            gate_scores = gate_scores * attention_mask

        # 3. Apply Gating (The Value Projection)
        gated_values = gate_scores.unsqueeze(-1) * self.value_proj(engram_embeddings)
        
        # 4. Global Top-K Selection across the sequence length L
        # K cannot exceed L
        k = min(self.k, L)
        topk_scores, topk_indices = torch.topk(gate_scores, k=k, dim=1)
        
        # Expand indices to gather the full D-dimensional embeddings
        gather_indices = topk_indices.unsqueeze(-1).expand(-1, -1, self.hidden_size)
        
        # Extract the final representation for late interaction
        final_authorship_representation = torch.gather(gated_values, dim=1, index=gather_indices)
        
        return final_authorship_representation, topk_indices


class DualMemoryLLMForAuthorshipVerification(ModernBertPreTrainedModel):
    config_class = DualMemoryLLMConfig
    _keys_to_ignore_on_load_missing = [
        r"authorship_head\..+",
        r"model\.layers\.\d+\.token_memory_mlp\..+",
        r"model\.layers\.\d+\.engram\..+",
    ]

    def __init__(self, config: DualMemoryLLMConfig):
        super().__init__(config)
        self.config = config
        
        self.model = DualMemoryLLMModel(config)
        
        engram_hidden_size = (config.max_ngram_size - 1) * config.n_embed_per_ngram if config.use_ngram_memory else config.hidden_size
        
        self.authorship_head = AuthorshipEngramHead(
            hidden_size=config.hidden_size,
            engram_hidden_size=engram_hidden_size, # Engram D
            k_selection=128
        )
        
        self.post_init()

    @torch.no_grad()
    def _init_weights(self, module):
        """
        Initialize the weights, specifically handling the new components.
        We call the base ModernBertPreTrainedModel._init_weights for standard parts
        and then manually handle our custom layers (Linear, Conv1d, RMSNorm, Embedding).
        """
        # Call base class init first for standard ModernBert submodules
        # We need to access the class method because 'super()._init_weights(module)' might not work in 'apply' 
        # context if not careful, but usually it does.
        # However, ModernBertPreTrainedModel._init_weights only handles specific ModernBert types.
        
        std = getattr(self.config, "initializer_range", 0.02)

        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.Conv1d):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, (nn.LayerNorm, nn.RMSNorm)):
            module.weight.data.fill_(1.0)
            if hasattr(module, "bias") and module.bias is not None:
                module.bias.data.zero_()
        
        # Now call the base class logic to handle things like Rotary Embedding and tokenizer embeddings
        # specifically if they match ModernBert types.
        try:
            super()._init_weights(module)
        except Exception:
            pass
        
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        **kwargs,
    ):
        
        # 1. Pass through Dual-Memory Transformer
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **kwargs,
        )
        
        last_hidden_state = outputs.last_hidden_state
        
        # 2. Fetch the corresponding engram embeddings used by the head
        if self.config.use_ngram_memory:
            engram_layer_id = self.config.engram_layer_ids[-1] if len(self.config.engram_layer_ids) > 0 else 0
            hash_input_ids = torch.from_numpy(self.model.hash_mapping.hash(input_ids.cpu().numpy())[engram_layer_id])
            hash_input_ids = hash_input_ids.to(last_hidden_state.device)
            
            # We need the multi_head_embedding logic for the engram_layer_id. We fetch it from the model layer
            layer = [l for l in self.model.layers if l.engram is not None and l.engram.layer_id == engram_layer_id]
            if not layer:
                # Fallback if no engram layer exists, though unlikely given config
                engram_vocab_sizes_for_layer = [x for y in self.model.hash_mapping.vocab_size_across_layers[engram_layer_id] for x in y]
                multi_head_emb = MultiHeadEmbedding(
                    list_of_N = engram_vocab_sizes_for_layer,
                    D = self.config.n_embed_per_ngram // self.config.n_head_per_ngram,
                ).to(last_hidden_state.device)
                engram_embs = multi_head_emb(hash_input_ids).flatten(start_dim=-2)
            else:
                engram_embs = layer[0].engram.multi_head_embedding(hash_input_ids).flatten(start_dim=-2)
        else:
            # Variant 2: Unigram memory
            # Use the initial token embeddings from the model
            engram_embs = self.model.embeddings(input_ids=input_ids, inputs_embeds=kwargs.get("inputs_embeds"))
            
        # 3. Pass through output head
        final_representation, topk_indices = self.authorship_head(last_hidden_state, engram_embs, attention_mask)
        
        return {
            "authorship_representation": final_representation,
            "topk_indices": topk_indices,
            "last_hidden_state": last_hidden_state
        }


if __name__ == "__main__":
    print("=== Testing Variant 1: Unigram Output (No Phrase-Level Memory) ===")
    config_unigram = DualMemoryLLMConfig(
        vocab_size=1000,
        pad_token_id=0,
        hidden_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=512,
        use_ngram_memory=False, # Disable N-Grams
        engram_layer_ids=[1, 3],
        engram_vocab_size=[1000, 1000],
        max_ngram_size=3,
        n_embed_per_ngram=64,
        n_head_per_ngram=2
    )

    model_unigram = DualMemoryLLMForAuthorshipVerification(config_unigram)
    input_ids = torch.randint(0, 100, (2, 32))
    outputs_unigram = model_unigram(input_ids=input_ids)

    print("Unigram Forward Pass Completed Successfully.")
    print(f"Final Authorship Representation Shape: {outputs_unigram['authorship_representation'].shape}")
    
    loss_unigram = outputs_unigram['authorship_representation'].sum()
    loss_unigram.backward()
    print("Unigram Backward Pass Completed Successfully.")
    
    has_attn_grad_u = model_unigram.model.layers[0].attn.Wqkv.weight.grad is not None
    has_token_mem_grad_u = model_unigram.model.layers[0].token_memory_mlp.Wi.weight.grad is not None
    print(f"Pathways received gradients -> Contextual: {has_attn_grad_u}, Token-Level: {has_token_mem_grad_u}")
    
    
    print("\n=== Testing Variant 2: Full N-Gram Output (With Phrase-Level Memory) ===")
    config_ngram = DualMemoryLLMConfig(
        vocab_size=1000,
        pad_token_id=0,
        hidden_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=512,
        use_ngram_memory=True, # Enable N-Grams
        engram_layer_ids=[1, 3],
        engram_vocab_size=[1000, 1000],
        max_ngram_size=3,
        n_embed_per_ngram=64,
        n_head_per_ngram=2
    )

    model_ngram = DualMemoryLLMForAuthorshipVerification(config_ngram)
    outputs_ngram = model_ngram(input_ids=input_ids)

    print("N-Gram Forward Pass Completed Successfully.")
    print(f"Final Authorship Representation Shape: {outputs_ngram['authorship_representation'].shape}")
    
    loss_ngram = outputs_ngram['authorship_representation'].sum()
    loss_ngram.backward()
    print("N-Gram Backward Pass Completed Successfully.")
    
    has_attn_grad_ng = model_ngram.model.layers[0].attn.Wqkv.weight.grad is not None
    has_token_mem_grad_ng = model_ngram.model.layers[0].token_memory_mlp.Wi.weight.grad is not None
    has_engram_grad_ng = model_ngram.model.layers[1].engram.value_proj.weight.grad is not None
    
    print(f"Pathways received gradients -> Contextual: {has_attn_grad_ng}, Phrase-Level: {has_engram_grad_ng}, Token-Level: {has_token_mem_grad_ng}")
