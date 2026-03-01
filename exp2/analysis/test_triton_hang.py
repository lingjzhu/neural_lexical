import torch
import sys
import os

# Add path for local modules
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from triton_kernels import fused_index_add
from fused_clustered_splade import ClusteredSpladeFusedMeanPooling
from clustered_splade import get_w_reduced

def test_triton():
    device = "cuda"
    num_clusters = 4000
    H = 1024
    V = 50368
    S = 512
    B = 32

    w = torch.randn(V, H, device=device, dtype=torch.bfloat16)
    cluster_ids = torch.randint(0, num_clusters, (V,), device=device, dtype=torch.long)
    h = torch.randn(B, S, H, device=device, dtype=torch.bfloat16)

    print("🔹 Testing fused_index_add...")
    w_reduced = fused_index_add(w, cluster_ids, num_clusters)
    torch.cuda.synchronize()
    print("✅ fused_index_add complete.")

    print("🔹 Testing get_w_reduced (with Triton)...")
    w_reduced_avg = get_w_reduced(w, cluster_ids, num_clusters, use_triton=True)
    torch.cuda.synchronize()
    print("✅ get_w_reduced complete.")

    print("🔹 Testing ClusteredSpladeFusedMeanPooling (with Triton)...")
    pooling = ClusteredSpladeFusedMeanPooling(num_clusters, activation="log1p_relu", use_triton=True).to(device)
    out = pooling(h, w, cluster_ids)
    torch.cuda.synchronize()
    print("✅ ClusteredSpladeFusedMeanPooling complete.")

if __name__ == "__main__":
    test_triton()
