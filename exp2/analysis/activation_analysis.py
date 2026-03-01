import torch
import torch.nn as nn
from transformers import AutoTokenizer
import json
import os
import sys
from tqdm import tqdm
from collections import Counter

# Add path for local modules
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Import the model classes defined in evaluate_hrs.py (they are evaluation-friendly)
from evaluate_hrs import ClusteredMLMTransformer, SparseEncoder

def main():
    checkpoint_path = "/home/slime-base/projects/jian/neural_lexical/exp2/outputs_clustered_modernbert_large_4000/checkpoint-1582"
    base_model = "answerdotai/ModernBERT-large"
    data_path = "/home/slime-base/projects/jian/neural_lexical/data/amazon_triplets.jsonl"
    num_samples = 1000
    num_clusters = 4000
    
    print(f"🔹 Loading model from {checkpoint_path}...")
    # Initialize the transformer. Note: evaluate_hrs version of ClusteredMLMTransformer 
    # handles the patching and cluster recovery in its constructor and recover_clusters_if_needed method.
    mlm = ClusteredMLMTransformer(
        checkpoint_path,
        max_seq_length=512,
        num_clusters=num_clusters,
        use_triton=True,
        activation="log1p_relu",
        model_type="modernbert"
    )
    model = SparseEncoder(modules=[mlm])
    
    # Recover clusters using base model weights
    mlm.recover_clusters_if_needed(base_model_path=base_model, model_dir=checkpoint_path)
    
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    if "modernbert" in checkpoint_path.lower():
        model.to(torch.bfloat16)
    print(f"🔹 Model moved to {model.device} and cast to bfloat16")

    print(f"🔹 Loading {num_samples} articles from {data_path}...")
    articles = []
    with open(data_path, "r") as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            obj = json.loads(line)
            articles.append(obj.get("positive", ""))

    cluster_counts = Counter()
    all_activations = [] # Store for visualization later

    print(f"🔹 Running inference on {len(articles)} articles...")
    tokenizer = mlm.tokenizer
    
    # Process in batches for speed
    batch_size = 32
    for i in tqdm(range(0, len(articles), batch_size)):
        batch_texts = articles[i:i+batch_size]
        features = tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
        features = {k: v.to(model.device) for k, v in features.items()}
        
        with torch.no_grad():
            with torch.autocast(device_type="cuda", enabled=torch.cuda.is_available(), dtype=torch.bfloat16):
                outputs = model(features)
                sparse_embeddings = outputs["sparse_embeddings"] # [Batch, num_clusters]
                
        # Boolean mask of active clusters (value > 0)
        active_mask = sparse_embeddings > 0
        active_indices = active_mask.nonzero(as_tuple=False)
        
        for idx in range(sparse_embeddings.shape[0]):
            # Get specific counts for this sample
            sample_active = active_indices[active_indices[:, 0] == idx][:, 1]
            cluster_counts.update(sample_active.tolist())
            
            # Save top 50 activations for the first 10 articles for later visualization
            if i + idx < 10:
                vals, inds = torch.topk(sparse_embeddings[idx], k=min(50, num_clusters))
                active_entries = []
                for v, ind in zip(vals, inds):
                    if v > 0:
                        active_entries.append({"cluster_id": int(ind), "score": float(v)})
                all_activations.append({
                    "article_index": i + idx,
                    "text_snippet": batch_texts[idx][:200] + "...",
                    "top_activations": active_entries
                })

    # Save results
    output_dir = os.path.join(CURRENT_DIR, "results")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save overall frequencies
    with open(os.path.join(output_dir, "frequencies.json"), "w") as f:
        # Convert Counter keys (torch.int64) to strings for JSON
        json.dump({str(k): v for k, v in cluster_counts.items()}, f, indent=2)
        
    # Save sample activations
    with open(os.path.join(output_dir, "sample_activations.json"), "w") as f:
        json.dump(all_activations, f, indent=2)

    print(f"✅ Analysis complete. Results saved in {output_dir}")

if __name__ == "__main__":
    main()
