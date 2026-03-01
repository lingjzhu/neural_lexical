import torch
import torch.nn as nn
from transformers import AutoTokenizer
import json
import os
import sys
from tqdm import tqdm
from collections import defaultdict

# Add path for local modules
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

# Import the model classes defined in evaluate_hrs.py
from evaluate_hrs import ClusteredMLMTransformer, SparseEncoder

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Analyze token-cluster overlap for SPLADE")
    parser.add_argument("--checkpoint_path", required=True, type=str, help="Path to the model checkpoint")
    parser.add_argument("--base_model", required=True, type=str, help="Base model for recovering cluster_ids")
    parser.add_argument("--data_path", default="/home/slime-base/projects/jian/neural_lexical/data/amazon_triplets.jsonl", type=str)
    parser.add_argument("--num_samples", default=100, type=int)
    parser.add_argument("--num_clusters", default=4000, type=int)
    parser.add_argument("--top_k_clusters", default=50, type=int)
    parser.add_argument("--model_type", default="modernbert", type=str)
    
    args = parser.parse_args()
    
    # Run main logic with args
    # (Extracting main logic into a function that takes args might be cleaner, 
    # but I'll just adapt the variables in the main script for now)
    
    checkpoint_path = args.checkpoint_path
    base_model = args.base_model
    data_path = args.data_path
    num_samples = args.num_samples
    num_clusters = args.num_clusters
    top_k_clusters = args.top_k_clusters
    model_type = args.model_type

    print(f"🔹 Loading model from {checkpoint_path}...")
    mlm = ClusteredMLMTransformer(
        checkpoint_path,
        max_seq_length=512,
        num_clusters=num_clusters,
        use_triton=True,
        activation="log1p_relu",
        model_type=model_type
    )
    model = SparseEncoder(modules=[mlm])
    
    # Recover clusters
    mlm.recover_clusters_if_needed(base_model_path=base_model, model_dir=checkpoint_path)
    
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    if "bert" not in model_type.lower():
        model.to(torch.bfloat16)

    # Build cluster to tokens mapping
    print("🔹 Building cluster-to-tokens mapping...")
    cluster_ids = mlm.cluster_ids.cpu().numpy()
    cluster_to_tokens = defaultdict(set)
    for token_id, cluster_id in enumerate(cluster_ids):
        cluster_to_tokens[int(cluster_id)].add(token_id)

    print(f"🔹 Loading {num_samples} articles from {data_path}...")
    articles = []
    with open(data_path, "r") as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            obj = json.loads(line)
            articles.append(obj.get("positive", ""))

    tokenizer = mlm.tokenizer
    results = []

    print(f"🔹 Analyzing overlap on {len(articles)} articles...")
    for idx, text in enumerate(tqdm(articles)):
        # 1. Get input token IDs
        inputs = tokenizer(text, truncation=True, max_length=512, return_tensors="pt")
        input_token_ids = set(inputs["input_ids"][0].tolist())
        
        # 2. Run inference
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            with torch.autocast(device_type="cuda" if "cuda" in device else "cpu", enabled=True, dtype=torch.bfloat16):
                outputs = model(inputs)
                sparse_embedding = outputs["sparse_embeddings"][0] # [num_clusters]

        # 3. Get top 50 active clusters
        vals, inds = torch.topk(sparse_embedding, k=min(top_k_clusters, num_clusters))
        
        overlap_analysis = []
        for score, cluster_id in zip(vals, inds):
            cluster_id = int(cluster_id)
            score = float(score)
            
            if score <= 0:
                continue
                
            # 4. Analyze overlap
            cluster_tokens = cluster_to_tokens[cluster_id]
            matching_tokens = input_token_ids.intersection(cluster_tokens)
            
            matching_token_texts = [tokenizer.decode([t]) for t in matching_tokens]
            
            overlap_analysis.append({
                "cluster_id": cluster_id,
                "score": score,
                "overlap_count": len(matching_tokens),
                "overlap_tokens": matching_token_texts
            })

        results.append({
            "article_index": idx,
            "text_preview": text[:100] + "...",
            "input_token_count": len(input_token_ids),
            "top_cluster_overlap": overlap_analysis
        })

    # Save results
    output_dir = os.path.join(CURRENT_DIR, "results")
    os.makedirs(output_dir, exist_ok=True)
    
    model_name = os.path.basename(os.path.normpath(checkpoint_path))
    if model_name.startswith("checkpoint"):
        parent = os.path.basename(os.path.dirname(os.path.normpath(checkpoint_path)))
        model_name = f"{parent}_{model_name}"
        
    output_path = os.path.join(output_dir, f"overlap_{model_name}.json")
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"✅ Analysis complete. Results saved in {output_path}")
