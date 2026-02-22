## Goal

Implement the training pipeline for SPLADE with clustered embeddings.

## Model

In SPLADE, the lexical embeddings are computed by multiplying the last hidden states H (batch_size, seq_len, hidden_dim) with the unembedding matrix W (vocab_size, hidden_dim). 
But the lexical embeddings are usually highly redundant. We can cluster them to reduce redundancy.

So I want to make a new model that can cluster the lexical embeddings. The final embeddings are the sum of the embeddings of the tokens in the same cluster: $E = HW^TC$, where $C$ is the cluster assignment matrix that should be binary. Before training, initialize this matrix with k-means clustering. Then during training, update the cluster assignments as embeddings are updated. Please implement this operation as fast as possible. 

Note that the log1P relu in SPLADE is applied after the matrix multiplication. So you can compute $W^TC$ first and multiply it with $H$. 

Implement a torch version first. Then write a Triton fused kernel to speed up. Note that you need to provide a benchmarking script to measreu the precision, speed and memory usage in both forward and backward pass, in fp32 and bf16. 

Below is an implementation and you can refer to it for reference. But be critical.
```
class UnembeddingCompressSparseFunction(torch.autograd.Function): @staticmethod def forward(ctx, h, w, cluster_ids, num_clusters): # h: [B, S, H] -> [BS, H] # w: [V, H] # cluster_ids: [V] dtype = h.dtype device = h.device B, S, H = h.shape V, _ = w.shape h_2d = h.reshape(-1, H) # 1. Reduce Weights: [V, H] -> [C, H] # Using float32 for reduction precision w_f32 = w.to(torch.float32) w_reduced = torch.zeros(num_clusters, H, device=device, dtype=torch.float32) w_reduced.index_add_(0, cluster_ids, w_f32) w_reduced = w_reduced.to(dtype) # 2. Matmul: [BS, H] @ [H, C] -> [BS, C] out_2d = torch.matmul(h_2d, w_reduced.t()) ctx.save_for_backward(h_2d, w, cluster_ids, w_reduced) ctx.num_clusters = num_clusters return out_2d.reshape(B, S, num_clusters) @staticmethod def backward(ctx, grad_output): h_2d, w, cluster_ids, w_reduced = ctx.saved_tensors num_clusters = ctx.num_clusters # grad_output: [B, S, C] -> [BS, C] BS, C = grad_output.shape[0] * grad_output.shape[1], grad_output.shape[2] grad_output_2d = grad_output.reshape(-1, C) # 1. grad_H = grad_O @ w_reduced [BS, C] @ [C, H] -> [BS, H] grad_h_2d = torch.matmul(grad_output_2d, w_reduced) # 2. grad_w_reduced = grad_O.T @ H [C, BS] @ [BS, H] -> [C, H] # Again, use float32 for intermediate if precision is high priority, # but here we use input dtype to match cuBLAS expectations. grad_w_reduced = torch.matmul(grad_output_2d.t(), h_2d) # 3. grad_W = grad_w_reduced[cluster_ids] [V, H] # This is the backward of the index_add operation. grad_w = grad_w_reduced[cluster_ids] # Restore h shape grad_h = grad_h_2d.reshape(ctx.saved_tensors[0].shape) # [BS, H] ? No, needs B, S. # Wait, h_2d was save. Restore B, S. # We can use ctx.saved_tensors[0] shape as reference but it's 2D. # We'll just infer from grad_output. B, S = grad_output.shape[0], grad_output.shape[1] return grad_h.reshape(B, S, -1), grad_w, None, None class UnembeddingCompressSparse(nn.Module): def __init__(self, num_clusters): super().__init__() self.num_clusters = num_clusters def forward(self, h, w, cluster_ids): return UnembeddingCompressSparseFunction.apply(h, w, cluster_ids, self.num_clusters) @torch.no_grad() def update_mask(self, w, w_reduced): """ Greedily re-assigns tokens to clusters based on similarity. Keep the mask binary by using argmax. Fast: Takes ~15ms for V=150k, C=8k. """ # Similarity between tokens and cluster prototypes: [V, H] @ [H, C] -> [V, C] # This is the most computationally heavy part of keeping it binary. # But it's just one wide matmul. sims = torch.matmul(w, w_reduced.t()) # Binary (Hard) Assignment: Pick the best cluster per token new_cluster_ids = sims.argmax(dim=-1) return new_cluster_ids
```

Please implement the training pipeline for SPLADE with clustered embeddings within the `exp2` directory. But you should make it possible to integrate it with the existing SPLADE training pipeline. You can refer to the existing SPLADE training pipeline for reference. 

