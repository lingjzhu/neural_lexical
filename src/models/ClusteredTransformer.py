"""Lexicon-based SPLADE and Sparse ColBERT for ModernBERT."""
from pathlib import Path

import torch
from torch import nn
from huggingface_hub import hf_hub_download

from src.models.MLMTransformer import MLMTransformer


def cluster_centroids(weight, cluster_ids, num_clusters):
    """Mean unembedding weights per cluster, accumulating in float32."""
    sums = torch.zeros(num_clusters, weight.shape[1], device=weight.device, dtype=torch.float32)
    sums = sums.index_add(0, cluster_ids, weight.float())
    counts = torch.bincount(cluster_ids, minlength=num_clusters).clamp_min(1)
    return (sums / counts[:, None]).to(weight.dtype)


def clustered_maxsim(queries, documents, chunk_size=8):
    """Mean MaxSim over nonzero query tokens; returns [queries, documents]."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    query_lengths = (queries.abs().sum(-1) > 1e-6).sum(-1).clamp_min(1)
    rows = []
    for q_start in range(0, len(queries), chunk_size):
        q = queries[q_start:q_start + chunk_size]
        columns = []
        for d_start in range(0, len(documents), chunk_size):
            d = documents[d_start:d_start + chunk_size]
            scores = torch.einsum("ash,bth->abst", q, d).amax(-1).sum(-1)
            columns.append(scores / query_lengths[q_start:q_start + len(q), None])
        rows.append(torch.cat(columns, dim=1))
    return torch.cat(rows, dim=0)


class ClusteredTransformer(MLMTransformer):
    """Project the MLM head's hidden states onto learned token clusters.

    Use ``load(path_or_hub_id)`` for released checkpoints. Set
    ``initialize_clusters=True`` only when starting a new training run.
    """
    config_keys = MLMTransformer.config_keys + [
        "num_clusters", "embedding_type", "activation", "use_triton", "add_special_token"
    ]

    def __init__(self, model_name_or_path, num_clusters=4000, embedding_type="splade",
                 activation="relu", use_triton=False, initialize_clusters=False,
                 cluster_ids_path=None, **kwargs):
        if embedding_type not in ("splade", "colbert"):
            raise ValueError("embedding_type must be 'splade' or 'colbert'")
        if activation not in ("relu", "log1p_relu"):
            raise ValueError("activation must be 'relu' or 'log1p_relu'")
        kwargs["backend"] = "torch"
        kwargs.setdefault("max_seq_length", 512)
        kwargs.setdefault("add_special_token", embedding_type == "splade")
        config_args = dict(kwargs.pop("config_args", None) or {})
        config_args["reference_compile"] = False
        super().__init__(model_name_or_path, config_args=config_args, **kwargs)
        if self.config.model_type != "modernbert":
            raise ValueError("The released lexicon models use ModernBERT")
        self.num_clusters = num_clusters
        self.embedding_type = embedding_type
        self.activation = activation
        self.use_triton = use_triton
        # Keep the MLM transform (dense + activation + norm), bypass vocabulary logits.
        self.auto_model.set_output_embeddings(nn.Identity())
        if initialize_clusters:
            from sklearn.cluster import MiniBatchKMeans
            labels = MiniBatchKMeans(n_clusters=num_clusters, batch_size=10000,
                                    n_init="auto", random_state=42).fit_predict(
                self.w.detach().float().cpu().numpy())
            ids = torch.tensor(labels, dtype=torch.long)
        else:
            if cluster_ids_path is not None:
                cluster_path = Path(cluster_ids_path)
            elif Path(model_name_or_path).is_dir():
                cluster_path = Path(model_name_or_path) / "cluster_ids.pt"
            else:
                model_args = kwargs.get("model_args") or {}
                cluster_path = hf_hub_download(
                    model_name_or_path, "cluster_ids.pt", cache_dir=kwargs.get("cache_dir"),
                    **{k: model_args[k] for k in ("token", "revision", "local_files_only", "subfolder")
                       if k in model_args})
            ids = torch.load(cluster_path, map_location="cpu", weights_only=True)
        ids = ids.to(device=self.w.device, dtype=torch.long)
        if ids.shape != (self.w.shape[0],) or ids.min() < 0 or ids.max() >= num_clusters:
            raise ValueError("Cluster assignments do not match the vocabulary/cluster count")
        self.register_buffer("cluster_ids", ids)

    @property
    def w(self):
        return self.auto_model.get_input_embeddings().weight

    @torch.no_grad()
    def update_clusters(self, chunk_size=1024):
        """Greedy dot-product reassignment, as in paper equations 7–8."""
        centroids = cluster_centroids(self.w, self.cluster_ids, self.num_clusters)
        labels = [self.w[i:i + chunk_size].matmul(centroids.T).argmax(-1)
                  for i in range(0, len(self.w), chunk_size)]
        self.cluster_ids.copy_(torch.cat(labels))

    def forward(self, features):
        inputs = {k: features[k] for k in ("input_ids", "attention_mask", "token_type_ids")
                  if k in features}
        hidden = self.auto_model(**inputs).logits
        mask = features["attention_mask"].unsqueeze(-1)
        centroids = cluster_centroids(self.w, self.cluster_ids, self.num_clusters).to(hidden.dtype)
        if self.embedding_type == "splade" and self.use_triton:
            from src.kernels.fused_sumpool import FeatureCountFunction
            pooled = FeatureCountFunction.apply(hidden * mask, centroids)
            pooled = pooled / mask.sum(1).clamp_min(1)
            # Preserve the fused training implementation's post-pooling log saturation.
            if self.activation == "log1p_relu":
                pooled = torch.log1p(pooled)
            embeddings = torch.log1p(pooled)
        else:
            logits = torch.relu(hidden @ centroids.T) * mask
            if self.embedding_type == "colbert":
                embeddings = torch.log1p(logits)
            else:
                pooled = logits.sum(1) / mask.sum(1).clamp_min(1)
                if self.activation == "log1p_relu":
                    pooled = torch.log1p(pooled)
                embeddings = torch.log1p(pooled)
        features["dense_embeddings"] = None
        features["sparse_embeddings"] = embeddings
        return features

    def get_sentence_embedding_dimension(self):
        return self.num_clusters

    def save(self, output_path, safe_serialization=True, **kwargs):
        super().save(output_path, safe_serialization=safe_serialization, **kwargs)
        torch.save(self.cluster_ids.detach().cpu(), Path(output_path) / "cluster_ids.pt")
