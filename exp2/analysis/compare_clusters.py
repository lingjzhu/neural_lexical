import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoConfig
import json
import os
import sys
from sklearn.cluster import MiniBatchKMeans
import numpy as np
from collections import Counter

# Add path for local modules
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

def get_initial_cluster_ids(base_model_path, num_clusters=4000):
    print(f"🔹 Loading base model {base_model_path} to reinitialize clusters (on CPU)...")
    print(f"🔹 Loading tokenizer for {base_model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    print(f"🔹 Loading config for {base_model_path}...")
    config = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)
    print(f"🔹 Loading model weights for {base_model_path} on CPU (this may take a minute)...")
    # Load on CPU
    model = AutoModelForMaskedLM.from_pretrained(base_model_path, config=config, trust_remote_code=True).to("cpu")
    print(f"✅ Model loaded.")
    
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
        
    print(f"🔹 Initializing {num_clusters} clusters using KMeans (random_state=42) on CPU...")
    w_np = w.detach().cpu().float().numpy()
    kmeans = MiniBatchKMeans(n_clusters=num_clusters, 
                             batch_size=10000, 
                             n_init="auto",
                             random_state=42)
    print(f"🔹 Fitting KMeans on {w_np.shape} embedding matrix...")
    kmeans.fit(w_np)
    print(f"🔹 KMeans fit complete. Extracting labels...")
    cluster_ids = torch.tensor(kmeans.labels_, dtype=torch.long)
    print("✅ KMeans initialization complete.")
    return cluster_ids, tokenizer

def main():
    base_model = "answerdotai/ModernBERT-large"
    checkpoint_path = "/home/slime-base/projects/jian/neural_lexical/exp2/outputs_clustered_modernbert_large_4000/checkpoint-1582"
    num_clusters = 4000
    
    # 1. Get initial mapping
    initial_ids, tokenizer = get_initial_cluster_ids(base_model, num_clusters)
    
    # 2. Get trained mapping
    trained_ids_path = os.path.join(checkpoint_path, "cluster_ids.pt")
    if not os.path.exists(trained_ids_path):
        print(f"❌ Trained cluster_ids.pt not found at {trained_ids_path}")
        return
    trained_ids = torch.load(trained_ids_path, map_location="cpu")
    
    # 3. Compare statistics
    total_tokens = len(initial_ids)
    changed_mask = initial_ids != trained_ids
    num_changed = changed_mask.sum().item()
    percent_changed = (num_changed / total_tokens) * 100
    
    print(f"\n📊 --- Cluster Evolution Statistics ---")
    print(f"Total tokens in vocab: {total_tokens}")
    print(f"Tokens that changed clusters: {num_changed} ({percent_changed:.2f}%)")
    print(f"Tokens that stayed in the same cluster: {total_tokens - num_changed} ({100 - percent_changed:.2f}%)")
    
    # 4. Detailed mapping generation
    vocab = tokenizer.get_vocab()
    inv_vocab = {v: k for k, v in vocab.items()}
    
    initial_mapping = {i: [] for i in range(num_clusters)}
    trained_mapping = {i: [] for i in range(num_clusters)}
    
    for token_id in range(total_tokens):
        if token_id in inv_vocab:
            token_text = inv_vocab[token_id]
            initial_mapping[int(initial_ids[token_id])].append(token_text)
            trained_mapping[int(trained_ids[token_id])].append(token_text)
            
    # Save mappings
    with open(os.path.join(CURRENT_DIR, "cluster_mapping_initial.json"), "w") as f:
        json.dump(initial_mapping, f, indent=2)
    with open(os.path.join(CURRENT_DIR, "cluster_mapping_trained.json"), "w") as f:
        json.dump(trained_mapping, f, indent=2)
        
    print(f"\n✅ Saved initial mapping to cluster_mapping_initial.json")
    print(f"✅ Saved trained mapping to cluster_mapping_trained.json")

    # 5. Evolution details: which clusters changed most?
    initial_counts = Counter(initial_ids.tolist())
    trained_counts = Counter(trained_ids.tolist())
    
    diffs = []
    for i in range(num_clusters):
        diffs.append({
            "cluster_id": i,
            "initial_size": initial_counts[i],
            "trained_size": trained_counts[i],
            "size_change": trained_counts[i] - initial_counts[i]
        })
    
    # Sort by absolute change
    diffs.sort(key=lambda x: abs(x["size_change"]), reverse=True)
    
    print(f"\n🔝 Top 10 Clusters by Size Change:")
    for d in diffs[:10]:
        print(f"Cluster {d['cluster_id']}: {d['initial_size']} -> {d['trained_size']} (Change: {d['size_change']})")

if __name__ == "__main__":
    main()
