import torch
import os
import glob

cluster_dir = "precomputed_clusters"

for f in sorted(glob.glob(os.path.join(cluster_dir, "*.pt"))):
    basename = os.path.basename(f)
    cluster_ids = torch.load(f, map_location="cpu")
    
    # Extract expected num_clusters from filename
    # e.g. modernbert_large_4000_clusters.pt -> 4000
    parts = basename.replace("_clusters.pt", "").split("_")
    expected_k = int(parts[-1])
    
    vocab_size = cluster_ids.shape[0]
    unique_ids = torch.unique(cluster_ids)
    num_unique = unique_ids.numel()
    max_id = cluster_ids.max().item()
    min_id = cluster_ids.min().item()
    
    # Check coverage
    all_expected = set(range(expected_k))
    assigned = set(unique_ids.tolist())
    empty_clusters = all_expected - assigned
    
    print(f"\n{'='*60}")
    print(f"File: {basename}")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Expected clusters: {expected_k}")
    print(f"  Unique cluster IDs: {num_unique}")
    print(f"  ID range: [{min_id}, {max_id}]")
    print(f"  Empty clusters: {len(empty_clusters)}")
    
    if empty_clusters:
        # Show distribution
        counts = torch.bincount(cluster_ids, minlength=expected_k)
        zero_count = (counts == 0).sum().item()
        min_count = counts[counts > 0].min().item()
        max_count = counts.max().item()
        mean_count = counts.float().mean().item()
        print(f"  ⚠️  {zero_count} clusters have ZERO tokens!")
        print(f"  Min tokens/cluster (non-empty): {min_count}")
        print(f"  Max tokens/cluster: {max_count}")
        print(f"  Mean tokens/cluster: {mean_count:.1f}")
        if len(empty_clusters) <= 20:
            print(f"  Empty cluster IDs: {sorted(empty_clusters)}")
    else:
        counts = torch.bincount(cluster_ids, minlength=expected_k)
        min_count = counts.min().item()
        max_count = counts.max().item()
        mean_count = counts.float().mean().item()
        print(f"  ✅ All clusters have at least one token!")
        print(f"  Min tokens/cluster: {min_count}")
        print(f"  Max tokens/cluster: {max_count}")
        print(f"  Mean tokens/cluster: {mean_count:.1f}")
