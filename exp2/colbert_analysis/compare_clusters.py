import torch
import torch.nn as nn
from transformers import AutoTokenizer
import json
import os
import sys
from collections import Counter

# Add path for local modules
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '../..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

def main():
    checkpoint_path = "/home/slimelab/Projects/neural_lexical/exp2/outputs_clustered_colbert_modernbert_large_2000_relu/checkpoint-1580"
    num_clusters = 2000
    initial_clusters_path = "/home/slimelab/Projects/neural_lexical/exp2/precomputed_clusters/modernbert_large_2000_clusters.pt"
    
    trained_clusters_path = os.path.join(checkpoint_path, "cluster_ids.pt")
    
    if not os.path.exists(initial_clusters_path):
        print(f"❌ Initial clusters not found at {initial_clusters_path}")
        return
    if not os.path.exists(trained_clusters_path):
        print(f"❌ Trained clusters not found at {trained_clusters_path}")
        return

    print(f"🔹 Loading initial clusters from {initial_clusters_path}...")
    initial_ids = torch.load(initial_clusters_path, map_location="cpu")
    
    print(f"🔹 Loading trained clusters from {trained_clusters_path}...")
    trained_ids = torch.load(trained_clusters_path, map_location="cpu")
    
    # Compare statistics
    total_tokens = len(initial_ids)
    changed_mask = initial_ids != trained_ids
    num_changed = changed_mask.sum().item()
    percent_changed = (num_changed / total_tokens) * 100
    
    print(f"\n📊 --- Cluster Evolution Statistics ---")
    print(f"Total tokens in vocab: {total_tokens}")
    print(f"Tokens that changed clusters: {num_changed} ({percent_changed:.2f}%)")
    print(f"Tokens that stayed in the same cluster: {total_tokens - num_changed} ({100 - percent_changed:.2f}%)")
    
    # Save statistics to JSON
    stats = {
        "total_tokens": total_tokens,
        "num_changed": num_changed,
        "percent_changed": percent_changed,
        "num_stable": total_tokens - num_changed,
        "percent_stable": 100 - percent_changed
    }
    
    output_path = os.path.join(CURRENT_DIR, "results", "cluster_evolution_stats.json")
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)
    
    # Evolution details: which clusters changed most?
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

    with open(os.path.join(CURRENT_DIR, "results", "cluster_size_changes.json"), "w") as f:
        json.dump(diffs, f, indent=2)

    print(f"\n✅ Comparison complete. Results saved in {os.path.join(CURRENT_DIR, 'results')}")

if __name__ == "__main__":
    main()
