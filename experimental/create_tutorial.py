import nbformat as nbf

nb = nbf.v4.new_notebook()

text0 = """# Dual-MemoryLLM Authorship Verification Tutorial

This notebook demonstrates how to initialize, configure, and run a forward/backward pass using the **Dual-MemoryLLM** model for Authorship Verification.

We will demonstrate two variants:
1. **Unigram Variant**: No Phrase-Level memory. Extracts features directly from token-level initial embeddings.
2. **N-Gram Variant**: The full model utilizing all 3 pathways (Contextual, Phrase-Level N-grams, Token-Level Memory).

Both variants will be initialized from `answerdotai/ModernBERT-base` using `from_pretrained()`."""

code1 = """import torch
from modeling_dual_memoryllm import DualMemoryLLMConfig, DualMemoryLLMForAuthorshipVerification

model_id = "answerdotai/ModernBERT-base\""""

text2 = """### 1. Unigram Variant (use_ngram_memory=False)
Initialize Dual-MemoryLLM bypassing N-gram tables."""

code2 = """print("=== Initializing Unigram Variant ===")
config_unigram = DualMemoryLLMConfig.from_pretrained(model_id)

# Configure for downstream logic
config_unigram.use_ngram_memory = False

# We ignore mismatched sizes during loading since we append our own custom heads and pathways
model_unigram = DualMemoryLLMForAuthorshipVerification.from_pretrained(
    model_id, 
    config=config_unigram, 
    ignore_mismatched_sizes=True
)

# Mock input batch [Batch_Size, Sequence_Length]
input_ids = torch.randint(1, 100, (2, 32)) 
outputs_unigram = model_unigram(input_ids=input_ids)

print("--- Forward Pass Results ---")
print(f"Final Authorship Representation Matrix: {outputs_unigram['authorship_representation'].shape}")
print(f"Top-K Engram Fingerprint Indices Shape: {outputs_unigram['topk_indices'].shape}")"""

text3 = """### 2. N-Gram Variant (use_ngram_memory=True)
Initialize the full Dual-MemoryLLM with Phrase-Level Memory."""

code3 = """print("=== Initializing Full N-Gram Variant ===")
config_ngram = DualMemoryLLMConfig.from_pretrained(model_id)

# Engram specialized configuration
config_ngram.use_ngram_memory = True
config_ngram.engram_layer_ids = [1, 3]          # which layers inject Phrase-Level Memory
config_ngram.engram_vocab_size = [1000, 1000]   # size of the n-gram hashing vocab tables
config_ngram.max_ngram_size = 3
config_ngram.n_embed_per_ngram = 64
config_ngram.n_head_per_ngram = 2

# We ignore mismatched sizes during loading since we append our own custom heads and pathways
model_ngram = DualMemoryLLMForAuthorshipVerification.from_pretrained(
    model_id, 
    config=config_ngram, 
    ignore_mismatched_sizes=True
)

outputs_ngram = model_ngram(input_ids=input_ids)

print("--- Forward Pass Results ---")
print(f"Final Authorship Representation Matrix: {outputs_ngram['authorship_representation'].shape}")
print(f"Top-K Engram Fingerprint Indices Shape: {outputs_ngram['topk_indices'].shape}")"""

text4 = """### 3. Demonstrating End-to-End Gradients
We demonstrate that both variants correctly participate in the backwards routing graph."""

code4 = """print("=== Backward Pass Checks ===")
loss_unigram = outputs_unigram["authorship_representation"].sum()
loss_unigram.backward()

has_attn_grad_u = model_unigram.model.layers[0].attn.Wqkv.weight.grad is not None
has_token_mem_grad_u = model_unigram.model.layers[0].token_memory_mlp.Wi.weight.grad is not None
print(f"Unigram Pathways received gradients: Contextual: {has_attn_grad_u}, Token-Level: {has_token_mem_grad_u}")

loss_ngram = outputs_ngram["authorship_representation"].sum()
loss_ngram.backward()

has_attn_grad_ng = model_ngram.model.layers[0].attn.Wqkv.weight.grad is not None
has_token_mem_grad_ng = model_ngram.model.layers[0].token_memory_mlp.Wi.weight.grad is not None
has_engram_grad_ng = model_ngram.model.layers[1].engram.value_proj.weight.grad is not None
print(f"N-Gram Pathways received gradients: Contextual: {has_attn_grad_ng}, Phrase-Level: {has_engram_grad_ng}, Token-Level: {has_token_mem_grad_ng}")"""

nb['cells'] = [
    nbf.v4.new_markdown_cell(text0),
    nbf.v4.new_code_cell(code1),
    nbf.v4.new_markdown_cell(text2),
    nbf.v4.new_code_cell(code2),
    nbf.v4.new_markdown_cell(text3),
    nbf.v4.new_code_cell(code3),
    nbf.v4.new_markdown_cell(text4),
    nbf.v4.new_code_cell(code4)
]

with open('tutorial_dual_memoryllm.ipynb', 'w') as f:
    nbf.write(nb, f)
