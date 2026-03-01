"""
Precompute clusters for LLaDA-8B-Base using base model weights directly.
Bypasses ClusteredMLMTransformer to avoid the LoRA overhead during clustering.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from transformers import AutoModelForCausalLM, AutoConfig

EXP2_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "GSAI-ML/LLaDA-8B-Base"
CLUSTER_SIZES = [4000]
OUT_DIR = os.path.join(EXP2_DIR, "precomputed_clusters")
os.makedirs(OUT_DIR, exist_ok=True)

print(f"Loading {MODEL_NAME} weights (bfloat16, CPU-mapped)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    dtype=torch.bfloat16,
    attn_implementation="eager",
    device_map="cpu",  # keep on CPU, we only need lm_head.weight
)

# Extract the output projection weight (tied to input embeddings)
# LLaDA uses model.transformer.ff_out (not lm_head)
w = None
for name, module in model.named_modules():
    if name.endswith("lm_head") and hasattr(module, "weight"):
        w = module.weight.detach().float()
        print(f"Found lm_head: {name}, shape={w.shape}")
        break
    if name == "model.transformer.ff_out" and hasattr(module, "weight"):
        w = module.weight.detach().float()
        print(f"Found ff_out: {name}, shape={w.shape}")
        break

if w is None:
    # Fallback: embed_tokens / word_embeddings
    for name, module in model.named_modules():
        if ("embed_tokens" in name or "word_embeddings" in name) and hasattr(module, "weight"):
            w = module.weight.detach().float()
            print(f"Found embedding: {name}, shape={w.shape}")
            break

assert w is not None, "Could not find lm_head or embed_tokens"

del model  # free RAM/VRAM before KMeans

# Move to GPU for KMeans
w = w.cuda()
print(f"Weight on GPU: {w.shape}, dtype={w.dtype}")

sys.path.insert(0, EXP2_DIR)
from clustered_splade import UnembeddingCompressSparse

for k in CLUSTER_SIZES:
    out_path = os.path.join(OUT_DIR, f"llada_8b_{k}_clusters.pt")
    if os.path.exists(out_path):
        print(f"Already exists: {out_path}")
        continue
    print(f"Computing {k} clusters for LLaDA-8B (vocab={w.shape[0]})...")
    compressor = UnembeddingCompressSparse(num_clusters=k)
    cluster_ids = compressor.init_kmeans(w)
    torch.save(cluster_ids.cpu(), out_path)
    print(f"Saved: {out_path}")

print("Done.")
