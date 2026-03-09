import torch
local_path = "/home/slimelab/Projects/neural_lexical/checkpoints/clustered_colbert_v1/cluster_ids.pt"
try:
    cluster_ids = torch.load(local_path, map_location="cpu")
    print(f"Loaded from {local_path}")
    print(f"Shape: {cluster_ids.shape}")
    print(f"Max: {cluster_ids.max().item()}, Min: {cluster_ids.min().item()}")
except Exception as e:
    print(e)
