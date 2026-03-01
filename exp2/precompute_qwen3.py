
import torch
import torch.nn as nn
import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train_clustered_splade import ClusteredMLMTransformer
from clustered_splade import UnembeddingCompressSparse

def precompute_qwen3():
    model_name = "Qwen/Qwen3-0.6B"
    model_type = "qwen3"
    num_clusters = 4000
    output_dir = "precomputed_clusters"
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, f"qwen3-0.6b_{num_clusters}_clusters.pt")
    
    print(f"Loading weights for {model_name}...")
    model_args = {"attn_implementation": "sdpa", "torch_dtype": torch.bfloat16}
    
    mlm = ClusteredMLMTransformer(
        model_name,
        model_type=model_type,
        model_args=model_args,
        num_clusters=1,
        max_seq_length=128
    )
    
    w = mlm.w
    print(f"Weights shape: {w.shape}")
    
    print(f"Computing {num_clusters} clusters for {model_name}...")
    compressor = UnembeddingCompressSparse(num_clusters=num_clusters)
    cluster_ids = compressor.init_kmeans(w)
    
    torch.save(cluster_ids.cpu(), output_path)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    precompute_qwen3()
