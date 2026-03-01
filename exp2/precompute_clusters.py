import torch
import torch.nn as nn
import os
import sys
from transformers import AutoModelForMaskedLM, AutoConfig

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train_clustered_splade import ClusteredMLMTransformer
from clustered_splade import UnembeddingCompressSparse

def extract_weights(model_name, model_type):
    print(f"Loading weights for {model_name}...")
    # Using the same logic as ClusteredMLMTransformer but without full module setup
    # We just need the weights for clustering
    
    model_args = {"attn_implementation": "sdpa"}
    if "bert" not in model_type.lower():
        model_args["torch_dtype"] = torch.bfloat16
        
    # We can use the ClusteredMLMTransformer's internal loading logic by just initializing it
    # with a dummy cluster count
    mlm = ClusteredMLMTransformer(
        model_name,
        model_type=model_type,
        model_args=model_args,
        num_clusters=1, # Dummy
        max_seq_length=128
    )
    return mlm.w

def precompute():
    models = [
        ("answerdotai/ModernBERT-large", "modernbert"),
        ("Qwen/Qwen3-0.6B", "qwen3"),
        ("dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1", "qwen3_diffusion"),
        ("GSAI-ML/LLaDA-8B-Base", "llada"),
    ]
    cluster_sizes = [1000, 2000, 4000, 8000]
    output_dir = "precomputed_clusters"
    os.makedirs(output_dir, exist_ok=True)
    
    for model_name, model_type in models:
        try:
            w = extract_weights(model_name, model_type)
            # Normalize weights for better clustering if needed? 
            # MiniBatchKMeans didn't normalize, let's stick to what we had.
            
            if model_type == "qwen3_diffusion":
                clean_name = "qwen3_0.6b_diffusion"
            elif model_type == "llada":
                clean_name = "llada_8b"
            else:
                clean_name = model_name.split('/')[-1].lower()
            
            for k in cluster_sizes:
                output_path = os.path.join(output_dir, f"{clean_name}_{k}_clusters.pt")
                if os.path.exists(output_path):
                    print(f"Skipping {output_path}, already exists.")
                    continue
                
                print(f"Computing {k} clusters for {model_name}...")
                compressor = UnembeddingCompressSparse(num_clusters=k)
                cluster_ids = compressor.init_kmeans(w)
                
                torch.save(cluster_ids.cpu(), output_path)
                print(f"Saved to {output_path}")
                
        except Exception as e:
            print(f"Failed for {model_name}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    precompute()
