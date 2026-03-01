import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoConfig
import json
import os
import sys
from sklearn.cluster import MiniBatchKMeans
import numpy as np

# Add path for local modules
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

def get_cluster_ids(checkpoint_path, base_model_path, num_clusters=4000):
    cluster_path = os.path.join(checkpoint_path, "cluster_ids.pt")
    if os.path.exists(cluster_path):
        print(f"🔹 Loading trained cluster_ids from {cluster_path}...")
        cluster_ids = torch.load(cluster_path, map_location="cpu").numpy()
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
        return cluster_ids, tokenizer

    print(f"⚠️ trained cluster_ids not found at {cluster_path}. Falling back to KMeans recovery from {base_model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    config = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained(base_model_path, config=config, trust_remote_code=True)
    
    # Find unembedding weight
    w = None
    for name, module in model.named_modules():
        if name.endswith("lm_head") or name.endswith("decoder") or name.endswith("ff_out") or name.endswith("head"):
            if hasattr(module, "weight"):
                if "ff_out" in name and "transformer.ff_out" not in name:
                    continue 
                w = module.weight
                break
    
    if w is None:
        for name, module in model.named_modules():
            if "tok_embeddings" in name or "word_embeddings" in name:
                if hasattr(module, "weight"):
                    w = module.weight
                    break
    
    if w is None:
        raise ValueError("Could not find unembedding weight")
        
    print(f"🔹 Initializing {num_clusters} clusters using KMeans (random_state=42)...")
    w_np = w.detach().cpu().float().numpy()
    kmeans = MiniBatchKMeans(n_clusters=num_clusters, 
                             batch_size=10000, 
                             n_init="auto",
                             random_state=42)
    kmeans.fit(w_np)
    cluster_ids = kmeans.labels_
    print("✅ KMeans complete.")
    return cluster_ids, tokenizer

def main():
    checkpoint_path = "/home/slime-base/projects/jian/neural_lexical/exp2/outputs_clustered_modernbert_large_4000/checkpoint-1582"
    base_model = "answerdotai/ModernBERT-large"
    num_clusters = 4000
    
    cluster_ids, tokenizer = get_cluster_ids(checkpoint_path, base_model, num_clusters)
    
    # Map cluster_id -> list of tokens
    mapping = {int(i): [] for i in range(num_clusters)}
    vocab = tokenizer.get_vocab()
    inv_vocab = {v: k for k, v in vocab.items()}
    
    for token_id, cluster_id in enumerate(cluster_ids):
        if token_id in inv_vocab:
            token_text = inv_vocab[token_id]
            mapping[int(cluster_id)].append(token_text)
            
    output_path = os.path.join(CURRENT_DIR, "cluster_mapping.json")
    with open(output_path, "w") as f:
        json.dump(mapping, f, indent=2)
    
    print(f"✅ Saved mapping to {output_path}")

if __name__ == "__main__":
    main()
