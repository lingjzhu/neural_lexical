import torch
import torch.nn as nn
from sklearn.cluster import MiniBatchKMeans
import numpy as np

try:
    from .triton_kernels import fused_index_add, fused_sim_argmax
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

class UnembeddingCompressSparseFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h, w, cluster_ids, num_clusters, use_triton=True):
        dtype = h.dtype
        device = h.device
        B, S, H = h.shape
        V, _ = w.shape
        h_2d = h.reshape(-1, H)
        
        # PyTorch native index_add_ in bf16 is mathematically safe, fast, and does not incur Triton 250MB JIT graph size
        # We avoid w.to(torch.float32) entirely here to save 400MB+ of peak memory allocation.
        w_reduced = torch.zeros(num_clusters, H, device=device, dtype=dtype)
        w_reduced.index_add_(0, cluster_ids, w)


        # Matmul: [BS, H] @ [H, C] -> [BS, C]
        out_2d = torch.matmul(h_2d, w_reduced.t())

        ctx.save_for_backward(h_2d, w, cluster_ids, w_reduced)
        ctx.num_clusters = num_clusters
        return out_2d.reshape(B, S, num_clusters)

    @staticmethod
    def backward(ctx, grad_output):
        h_2d, w, cluster_ids, w_reduced = ctx.saved_tensors
        num_clusters = ctx.num_clusters

        # grad_output: [B, S, C] -> [BS, C]
        BS, C = grad_output.shape[0] * grad_output.shape[1], grad_output.shape[2]
        grad_output_2d = grad_output.reshape(-1, C)

        # 1. grad_H = grad_O @ w_reduced -> [BS, C] @ [C, H] -> [BS, H]
        grad_h_2d = torch.matmul(grad_output_2d, w_reduced)

        # 2. grad_w_reduced = grad_O.T @ H -> [C, BS] @ [BS, H] -> [C, H]
        grad_w_reduced = torch.matmul(grad_output_2d.t(), h_2d)

        # 3. grad_W = grad_w_reduced[cluster_ids] -> [V, H]
        grad_w = grad_w_reduced[cluster_ids]

        # Restore h shape
        B, S = grad_output.shape[0], grad_output.shape[1]
        grad_h = grad_h_2d.reshape(B, S, -1)

        return grad_h, grad_w, None, None, None


class UnembeddingCompressSparse(nn.Module):
    def __init__(self, num_clusters, use_triton=True):
        super().__init__()
        self.num_clusters = num_clusters
        self.use_triton = use_triton

    def forward(self, h, w, cluster_ids):
        return UnembeddingCompressSparseFunction.apply(h, w, cluster_ids, self.num_clusters, self.use_triton)

    @torch.no_grad()
    def update_mask(self, w, w_reduced):
        """
        Greedily re-assigns tokens to clusters based on similarity.
        Keep the mask binary by using argmax.
        """
        if self.use_triton and HAS_TRITON:
            new_cluster_ids = fused_sim_argmax(w, w_reduced)
        else:
            sims = torch.matmul(w, w_reduced.t())
            new_cluster_ids = sims.argmax(dim=-1)
        return new_cluster_ids

    @torch.no_grad()
    def update_mask_faiss(self, w, niter=10):
        """
        Re-cluster tokens using FAISS KMeans (full E-M).
        Guarantees no empty clusters (FAISS re-seeds empty ones).
        Returns new cluster_ids on the same device as w.
        """
        assert HAS_FAISS, "FAISS is required for update_mask_faiss"
        w_np = w.detach().cpu().float().numpy()
        d = w_np.shape[1]
        
        use_gpu = False
        try:
            if faiss.get_num_gpus() > 0:
                use_gpu = True
        except AttributeError:
            pass
        
        kmeans = faiss.Kmeans(d, self.num_clusters, niter=niter, verbose=False, gpu=use_gpu)
        kmeans.train(w_np)
        _, labels = kmeans.index.search(w_np, 1)
        return torch.tensor(labels.flatten(), device=w.device, dtype=torch.long)

    @torch.no_grad()
    def init_kmeans(self, w):
        """
        Initialize cluster_ids using KMeans on the weights w.
        """
        print(f"Initializing {self.num_clusters} clusters using KMeans (FAISS={HAS_FAISS})...")
        w_np = w.detach().cpu().float().numpy()
        
        if HAS_FAISS:
            d = w_np.shape[1]
            # Detect GPU support safely
            use_gpu = False
            try:
                if faiss.get_num_gpus() > 0:
                    use_gpu = True
            except AttributeError:
                pass
            
            kmeans = faiss.Kmeans(d, self.num_clusters, niter=20, verbose=True, gpu=use_gpu)
            kmeans.train(w_np)
            _, labels = kmeans.index.search(w_np, 1)
            cluster_ids = torch.tensor(labels.flatten(), device=w.device, dtype=torch.long)
        else:
            kmeans = MiniBatchKMeans(n_clusters=self.num_clusters, 
                                     batch_size=10000, 
                                     n_init="auto",
                                     random_state=42)
            kmeans.fit(w_np)
            cluster_ids = torch.tensor(kmeans.labels_, device=w.device, dtype=torch.long)
            
        print("KMeans initialization complete.")
        return cluster_ids

def get_w_reduced(w: torch.Tensor, cluster_ids: torch.Tensor, num_clusters: int, use_triton=True) -> torch.Tensor:
    if use_triton and HAS_TRITON:
        w_reduced = fused_index_add(w, cluster_ids, num_clusters, dtype=torch.float32)
    else:
        w_f32 = w.to(torch.float32)
        w_reduced = torch.zeros(num_clusters, w.shape[1], device=w.device, dtype=torch.float32)
        w_reduced = w_reduced.index_add(0, cluster_ids, w_f32)
    
    # Compute the number of tokens assigned to each cluster
    counts = torch.bincount(cluster_ids, minlength=num_clusters).to(w_reduced.dtype)
    counts = counts.unsqueeze(1).clamp(min=1.0) # Avoid division by zero
    
    # Average the weights
    w_reduced = w_reduced / counts
    return w_reduced.to(w.dtype)

