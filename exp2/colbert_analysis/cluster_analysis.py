import torch
import torch.nn as nn
from transformers import AutoTokenizer
import json
import os
import sys

# Add path for local modules
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '../..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

def main():
    checkpoint_path = "/home/slimelab/Projects/neural_lexical/exp2/outputs_clustered_colbert_modernbert_large_2000_relu/checkpoint-1580"
    num_clusters = 2000
    
    cluster_path = os.path.join(checkpoint_path, "cluster_ids.pt")
    if not os.path.exists(cluster_path):
        print(f"❌ Error: cluster_ids.pt not found at {cluster_path}")
        return

    print(f"🔹 Loading trained cluster_ids from {cluster_path}...")
    cluster_ids = torch.load(cluster_path, map_location="cpu").numpy()
    
    # Load tokenizer from checkpoint
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
    
    # Map cluster_id -> list of tokens
    mapping = {int(i): [] for i in range(num_clusters)}
    vocab = tokenizer.get_vocab()
    inv_vocab = {v: k for k, v in vocab.items()}
    
    for token_id, cluster_id in enumerate(cluster_ids):
        if token_id in inv_vocab:
            token_text = inv_vocab[token_id]
            mapping[int(cluster_id)].append(token_text)
            
    output_path = os.path.join(CURRENT_DIR, "results", "cluster_mapping.json")
    with open(output_path, "w") as f:
        json.dump(mapping, f, indent=2)
    
    print(f"✅ Saved mapping of {len(cluster_ids)} tokens to {num_clusters} clusters at {output_path}")

if __name__ == "__main__":
    main()
