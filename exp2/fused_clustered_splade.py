import sys
import os
import torch
import torch.nn as nn

try:
    from clustered_splade import get_w_reduced
except ImportError:
    from .clustered_splade import get_w_reduced

# Import the existing highly optimized fused sum pooling kernel
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
try:
    from kernels.fused_sumpool import FeatureCountFunction
    HAS_FUSED_KERNEL = True
except ImportError:
    HAS_FUSED_KERNEL = False

class ClusteredSpladeFusedMeanPooling(nn.Module):
    """
    Fuses Unembedding + Activation + Mean Pooling.
    
    Instead of projecting the full sequence to the vocabulary dimension and then pooling,
    this module reduces the unembedding weights to the cluster dimension first, and 
    then uses a memory-efficient fused Triton kernel (FeatureCountFunction) to compute 
    activations and pooling without materializing intermediate [Batch, Seq, Cluster] logits.
    """
    def __init__(self, num_clusters, activation="relu", use_triton=True):
        super().__init__()
        self.num_clusters = num_clusters
        self.activation = activation
        self.use_triton = use_triton
        
        if self.activation == "log1p_relu" and self.use_triton:
            import logging
            logging.warning("FeatureCountFunction natively computes ReLU sums. "
                            "For log1p_relu, we apply log1p AFTER sum pooling as an approximation.")
        
        self.register_buffer("_cached_w_reduced", None, persistent=False)
        self._cached_w_version = -1
        
    def forward(self, h, w, cluster_ids, attention_mask=None):
        B, S, D = h.shape
        
        # 1. Precompute W_reduced [C, D] with caching
        # We use w._version to detect if weights have changed.
        # CRITICAL: If the cached tensor was created in inference mode (e.g. during eval),
        # and we are now in training mode, we MUST recompute it so it has a gradient path.
        current_w_version = getattr(w, "_version", 0)
        is_training = torch.is_grad_enabled()
        
        should_recompute = (
            self._cached_w_reduced is None or 
            self._cached_w_version != current_w_version or
            (is_training and self._cached_w_reduced.is_inference())
        )

        if not should_recompute:
            w_reduced = self._cached_w_reduced
        else:
            w_reduced = get_w_reduced(w, cluster_ids, self.num_clusters, self.use_triton)
            self._cached_w_reduced = w_reduced
            self._cached_w_version = current_w_version

        # Ensure w_reduced matches h's dtype (e.g. if w was left as fp32 while h is bf16)
        w_reduced = w_reduced.to(h.dtype)

        # 2. Fused Sum Pooling
        if self.use_triton and HAS_FUSED_KERNEL and self.activation in ["relu", "log1p_relu"]:
            # FeatureCountFunction does: ReLU(X @ W.T).sum(dim=1)
            # Memory layout: no intermediate [B, S, C] tensor is materialized
            total_sum = FeatureCountFunction.apply(h, w_reduced)
        else:
            # PyTorch fallback (materializes intermediate [B, S, C] logits)
            logits = torch.matmul(h, w_reduced.t())
            if self.activation == "relu":
                logits = torch.relu(logits)
            elif self.activation == "log1p_relu":
                logits = torch.log1p(torch.relu(logits))
            total_sum = torch.sum(logits, dim=1)
        
        # 3. Mean Pooling Normalization
        if attention_mask is not None:
            # Ensure denominator is at least 1 to avoid DivisionByZero
            counts = attention_mask.sum(dim=1).clamp_min(1).unsqueeze(-1)
            pooled = total_sum / counts
        else:  
            pooled = total_sum / S

        # 4. Final Activations (Replacing LightSpladePooling functionality)
        if self.use_triton and HAS_FUSED_KERNEL and self.activation == "log1p_relu":
            # For Triton with log1p_relu, we missed the inner log1p, so we apply an extra one outside
            pooled = torch.log1p(torch.log1p(pooled))
        else:
            # Standard: if relu, we apply the final log1p. 
            # If PyTorch with log1p_relu, the inner log1p was already applied, so we just apply the final log1p.
            pooled = torch.log1p(pooled)
            
        return pooled

