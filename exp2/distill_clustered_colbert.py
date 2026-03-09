import sys
import os
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
os.environ["HF_TRUST_REMOTE_CODE"] = "1"
os.environ["TRUST_REMOTE_CODE"] = "True"

import sys
import importlib.machinery
from unittest.mock import MagicMock

m = MagicMock()
m.__spec__ = importlib.machinery.ModuleSpec("flash_attn", None)

sys.modules['flash_attn'] = m
sys.modules['flash_attn.flash_attn_interface'] = m
sys.modules['flash_attn.layers'] = m
sys.modules['flash_attn.layers.rotary'] = m
sys.modules['flash_attn.ops'] = m
sys.modules['flash_attn.ops.triton'] = m
sys.modules['flash_attn.ops.triton.rotary'] = m
sys.modules['flash_attn_2_cuda'] = m

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
from typing import Iterable
from datasets import load_dataset
from sentence_transformers import SparseEncoderTrainer, SparseEncoderTrainingArguments

from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from src.models.MLMTransformer import MLMTransformer
from src.loss.SpladeLoss import SparseSelfMultipleNegativesRankingLoss, CachedSpladeMixedTopKLoss
from src.models.SparseColbertEncoder import SparseColbertEncoder
from src.models.sparse_colbert_triplet import SparseColBERTTripletEvaluator
from src.loss.SpladeLoss import SpladeMixedTopKLoss
from src.loss.SpladeLoss import gather_tensor, _cached_splade_backward_hook, CachedSpladeMixedTopKLoss, SpladeMixedTopKLoss
from functools import partial



def memory_efficient_colbert_similarity(q, d, chunk_size=8):
    '''
    Memory efficient similarity function for computing pairwise ColBERT scores across batches.
    Avoids constructing the massive similarity matrix that leads to OOM in standard PyTorch implementations.
    '''
    q = q.to(torch.bfloat16)
    d = d.to(torch.bfloat16)
    
    B1 = q.shape[0]
    q_mask = (q.abs().sum(dim=-1) > 1e-6).float()
    qlen = q_mask.sum(dim=1).clamp(min=1.0)
    
    out_chunks = []
    for i in range(0, B1, chunk_size):
        q_chunk = q[i:i+chunk_size] # [chunk_size, S_q, C]
        # [chunk_size, S_q, C] x [B2, S_d, C] -> [chunk_size, B2, S_q, S_d]
        chunk_scores = torch.einsum("ash,bth->abst", q_chunk, d)
        chunk_max = chunk_scores.max(dim=-1).values # [chunk_size, B2, S_q]
        chunk_sum = chunk_max.sum(dim=-1) # [chunk_size, B2]
        chunk_sum = chunk_sum / qlen[i:i+chunk_size].unsqueeze(1).to(chunk_sum.dtype)
        out_chunks.append(chunk_sum)
        
    return torch.cat(out_chunks, dim=0)


class ClusteredColbertTransformer(MLMTransformer):
    def __init__(self, *args, num_clusters=8000, use_triton=True, activation="relu", model_type="modern-bert", scale_embeddings=True, unfreeze_embeddings=False, use_fp8=False, **kwargs):
        if "modern" in model_type.lower() or "bert" in model_type.lower():
            kwargs["backend"] = "torch"
        super().__init__(*args, **kwargs)
        self.num_clusters = num_clusters
        self.model_type = model_type
        self.scale_embeddings = scale_embeddings
        self.unfreeze_embeddings = unfreeze_embeddings
        self.use_fp8 = use_fp8
        self.activation = activation
        self.use_triton = use_triton
        
        if hasattr(self.auto_model.config, "reference_compile"):
            self.auto_model.config.reference_compile = False
        
        self.w = None
        for name, module in self.auto_model.named_modules():
            if name.endswith("lm_head") or name.endswith("decoder") or name.endswith("ff_out"):
                if hasattr(module, "weight"):
                    if "ff_out" in name and "transformer.ff_out" not in name:
                        continue 
                    self.w = module.weight
                    if hasattr(self.w, "modules_to_save"):
                        self.w = self.w.default.weight
                    
                    parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
                    if parent_name:
                        parent = self.auto_model.get_submodule(parent_name)
                        setattr(parent, name.split('.')[-1], nn.Identity())
                    else:
                        setattr(self.auto_model, name, nn.Identity())
                    break
        
        if self.w is None:
            if hasattr(self.auto_model, "get_output_embeddings") and hasattr(self.auto_model.get_output_embeddings(), "weight"):
                w_mod = self.auto_model.get_output_embeddings()
                self.w = w_mod.weight
                if hasattr(self.auto_model, "set_output_embeddings"):
                    self.auto_model.set_output_embeddings(nn.Identity())
            else:
                raise ValueError("Could not find lm_head or decoder in auto_model through named_modules")

        from clustered_splade import UnembeddingCompressSparse
        temp_layer = UnembeddingCompressSparse(num_clusters, use_triton)
        
        model_name_or_path = args[0]
        local_cluster_path = os.path.join(model_name_or_path, "cluster_ids.pt")
        
        cluster_loaded_successfully = False
        if os.path.exists(local_cluster_path):
            print(f"Loading cluster IDs from local checkpoint: {local_cluster_path}")
            cluster_ids_loaded = torch.load(local_cluster_path, map_location=self.w.device)
            if cluster_ids_loaded.shape[0] == self.w.shape[0] and cluster_ids_loaded.max().item() < num_clusters:
                cluster_ids_val = cluster_ids_loaded
                cluster_loaded_successfully = True
            else:
                print(f"Warning: Local cluster IDs shape/max invalid for requested num_clusters={num_clusters}. Falling back.")
                
        if not cluster_loaded_successfully:
            path_lower = model_name_or_path.lower()
            if "modernbert-large" in path_lower or "modernbert_large" in path_lower:
                model_key = "modernbert_large"
            elif "modernbert-base" in path_lower or "modernbert_base" in path_lower or "modernbert" in path_lower:
                model_key = "modernbert"
            else:
                model_key = model_name_or_path.rstrip('/').split('/')[-1].replace('-', '_')
                
            precomputed_path = os.path.join(os.path.dirname(__file__), "precomputed_clusters", f"{model_key}_{num_clusters}_clusters.pt")
            
            if os.path.exists(precomputed_path):
                print(f"Loading precomputed clusters from {precomputed_path}...")
                cluster_ids_val = torch.load(precomputed_path, map_location=self.w.device)
            else:
                print(f"No local or precomputed clusters found (searched {local_cluster_path} and {precomputed_path}). Initializing with KMeans...")
                cluster_ids_val = temp_layer.init_kmeans(self.w)
            
        self.cluster_ids = nn.Parameter(cluster_ids_val, requires_grad=False)
        self._cached_w_reduced = None
        self._cached_w_version = -1

    def forward(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        trans_features = {
            key: value
            for key, value in features.items()
            if key in ["input_ids", "attention_mask", "token_type_ids"]
        }

        outputs = self.auto_model(**trans_features)
        
        try:
            if hasattr(outputs, "logits"):
                hidden_states = outputs.logits
            else:
                hidden_states = outputs.last_hidden_state
            
            if hidden_states.dim() == 2:
                hidden_states = hidden_states.unsqueeze(1) # [B, 1, H]
            
            if self.scale_embeddings:
                d = hidden_states.shape[-1]
                hidden_states = hidden_states * (d ** 0.5)
                
            from clustered_splade import get_w_reduced
            current_w_version = getattr(self.w, "_version", 0)
            is_training = torch.is_grad_enabled()
            
            should_recompute = (
                self._cached_w_reduced is None or 
                self._cached_w_version != current_w_version or
                is_training  # Always recompute during training to generate fresh computation graphs per-minibatch
            )

            if not should_recompute:
                w_reduced = self._cached_w_reduced
            else:
                w_reduced = get_w_reduced(self.w, self.cluster_ids, self.num_clusters, self.use_triton)
                if not is_training:
                    self._cached_w_reduced = w_reduced
                    self._cached_w_version = current_w_version

            w_reduced = w_reduced.to(hidden_states.dtype)
            
            # Matmul: [B, S, H] @ [H, C] -> [B, S, C]
            clustered_logits = torch.matmul(hidden_states, w_reduced.t())

            if self.activation == "relu":
                clustered_logits = torch.relu(clustered_logits)
            elif self.activation == "log1p_relu":
                clustered_logits = torch.log1p(torch.relu(clustered_logits))
                
            if self.activation == "relu":
                clustered_logits = torch.log1p(clustered_logits)
                
            attention_mask = features.get("attention_mask")
            if attention_mask is not None:
                clustered_logits = clustered_logits * attention_mask.unsqueeze(-1)
            
            S = clustered_logits.shape[1]
            if S < 512:
                # Use F.pad for more memory-efficient padding (avoids large intermediate zeros tensor)
                # pad format: (left, right, top, bottom, front, back) for last 3 dims
                # we want to pad the sequence dimension (dim 1) at the end
                # (pad_dim_2_left, pad_dim_2_right, pad_dim_1_left, pad_dim_1_right, pad_dim_0_left, pad_dim_0_right)
                clustered_logits = F.pad(clustered_logits, (0, 0, 0, 512 - S))
                
            features["sparse_embeddings"] = clustered_logits.to(torch.bfloat16)
            # Remove redundant dense_embeddings storage to save memory during inference/eval
            # features["dense_embeddings"] = outputs.hidden_states if hasattr(outputs, 'hidden_states') else None
            
        except AttributeError:
            features = None

        return features

    def get_sentence_embedding_dimension(self) -> int:
        return self.num_clusters




class CachedDistillationLoss(CachedSpladeMixedTopKLoss):
    def __init__(self, mse_weight=1.0, mnrl_weight=1.0, teacher_score_scale=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mse_weight = mse_weight
        self.mnrl_weight = mnrl_weight
        self.teacher_score_scale = teacher_score_scale

    def calculate_loss_and_cache_gradients(self, reps: list[list[torch.Tensor]], labels: torch.Tensor | None = None) -> torch.Tensor:
        loss = self.calculate_contrastive_loss(reps, labels=labels, with_backward=True)
        if loss.requires_grad:
            loss = loss.detach().requires_grad_()
        self.cache = [[r.grad for r in rs] for rs in reps]
        return loss

    def calculate_contrastive_loss(self, reps: list[list[torch.Tensor]], labels: torch.Tensor | None = None, with_backward: bool = False) -> torch.Tensor:
        anchors = gather_tensor(torch.cat(reps[0], dim=0))
        candidates_list = [gather_tensor(torch.cat(r, dim=0)) for r in reps[1:]] 
        candidates = torch.cat(candidates_list, dim=0)
        
        batch_size = anchors.size(0)
        num_candidates = len(candidates_list)
        idx = torch.arange(batch_size, device=anchors.device)

        q_candidates = torch.cat([candidates, anchors], dim=0)
        q_scores = self.sparse_loss.similarity_fct(anchors, q_candidates) * self.sparse_loss.scale
        q_scores[idx, num_candidates * batch_size + idx] = float('-inf')
        loss_q = self.sparse_loss.cross_entropy_loss(q_scores, idx)

        positives = candidates_list[0]
        d_candidates = torch.cat([anchors, candidates], dim=0)
        d_scores = self.sparse_loss.similarity_fct(positives, d_candidates) * self.sparse_loss.scale
        d_scores[idx, batch_size + idx] = float('-inf')
        loss_d = self.sparse_loss.cross_entropy_loss(d_scores, idx)

        loss_mnrl = (loss_q + loss_d) / 2

        loss_mse = 0.0
        if labels is not None and num_candidates >= 2:
            labels_gathered = gather_tensor(labels)
            
            pos_scores = q_scores[idx, idx]
            neg_scores = q_scores[idx, batch_size + idx]
            
            student_margin = pos_scores - neg_scores
            
            teacher_margin = labels_gathered.to(student_margin.dtype)
            mean = teacher_margin.mean()
            std = teacher_margin.std()
            if torch.isnan(std) or std == 0.0:
                std = 1.0
            teacher_margin_norm = (teacher_margin - mean) / (std + 1e-8) * self.teacher_score_scale
            
            loss_mse = torch.nn.functional.mse_loss(student_margin, teacher_margin_norm)
        total_loss = self.mnrl_weight * loss_mnrl + self.mse_weight * loss_mse

        if with_backward:
            if not total_loss.requires_grad:
                total_loss = total_loss.requires_grad_()
            total_loss.backward()
            total_loss = total_loss.detach().requires_grad_()
            
        self.last_mnrl_loss = loss_mnrl.item()
        self.last_mse_loss = loss_mse.item()
        return total_loss

    def forward(
        self, sentence_features: Iterable[dict[str, torch.Tensor]], labels: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        reps = []
        self.random_states = []
        
        for sentence_feature in sentence_features:
            reps_mbs = []
            random_state_mbs = []
            for reps_mb_dict, random_state in self.embed_minibatch_iter(
                sentence_feature=sentence_feature,
                with_grad=False,
                copy_random_state=True
            ):
                sparse_emb = reps_mb_dict["sparse_embeddings"].detach().requires_grad_()
                reps_mbs.append(sparse_emb)
                random_state_mbs.append(random_state)
            reps.append(reps_mbs)
            self.random_states.append(random_state_mbs)

        losses = {}
        current_reg_w = self._get_regularizer_weight()

        if torch.is_grad_enabled():
            loss = self.calculate_loss_and_cache_gradients(reps, labels=labels)
            loss.register_hook(partial(_cached_splade_backward_hook, sentence_features=sentence_features, loss_obj=self))
            losses["sparse_loss"] = loss
            
            if self.use_document_regularizer_only:
                corpus_loss = sum(self.calculate_reg_loss_no_grad(reps, i) for i in range(len(reps))) / len(reps)
                losses["document_regularizer_loss"] = corpus_loss * current_reg_w
            else:
                corpus_loss = sum(self.calculate_reg_loss_no_grad(reps, i) for i in range(1, len(reps))) / max(1, len(reps)-1)
                losses["document_regularizer_loss"] = corpus_loss * current_reg_w
                if self.query_regularizer_weight is not None:
                    query_loss = self.calculate_reg_loss_no_grad(reps, 0)
                    losses["query_regularizer_loss"] = query_loss * current_reg_w
        else:
            losses["sparse_loss"] = self.calculate_contrastive_loss(reps, labels=labels)
            if self.use_document_regularizer_only:
                corpus_loss = sum(self.calculate_reg_loss_no_grad(reps, i) for i in range(len(reps))) / len(reps)
                losses["document_regularizer_loss"] = corpus_loss * current_reg_w
            else:
                corpus_loss = sum(self.calculate_reg_loss_no_grad(reps, i) for i in range(1, len(reps))) / max(1, len(reps)-1)
                losses["document_regularizer_loss"] = corpus_loss * current_reg_w
                if self.query_regularizer_weight is not None:
                    query_loss = self.calculate_reg_loss_no_grad(reps, 0)
                    losses["query_regularizer_loss"] = query_loss * current_reg_w

        if self.sparse_loss is not None:
            self.sparse_loss.scale = self._get_scale()
            self._step += 1
        
        self._reg_step += 1

        return losses

class DistillationLoss(SpladeMixedTopKLoss):
    def __init__(self, mse_weight=1.0, mnrl_weight=1.0, teacher_score_scale=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mse_weight = mse_weight
        self.mnrl_weight = mnrl_weight
        self.teacher_score_scale = teacher_score_scale

    def forward(
        self, sentence_features: Iterable[dict[str, torch.Tensor]], labels: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        embeddings = [self.model(sentence_feature) for sentence_feature in sentence_features]
        sparse_embeddings = [embedding["sparse_embeddings"] for embedding in embeddings]
        dense_embeddings = [embedding.get("dense_embeddings") for embedding in embeddings]

        losses = {}
        base_loss = 0.0

        current_reg_w = self._get_regularizer_weight()

        if self.use_document_regularizer_only:
            corpus_loss = self.document_regularizer.compute_loss_from_embeddings(torch.cat(sparse_embeddings))
        else:
            corpus_loss = self.document_regularizer.compute_loss_from_embeddings(torch.cat(sparse_embeddings[1:]))
        losses["document_regularizer_loss"] = corpus_loss * current_reg_w

        if self.query_regularizer_weight is not None:
            query_loss = self.query_regularizer.compute_loss_from_embeddings(sparse_embeddings[0])
            losses["query_regularizer_loss"] = query_loss * current_reg_w

        if self.sparse_loss is not None:
            if self.logit_soft_capping_value > 0.0:
                sparse_embeddings = [self.logit_soft_capping(embedding, temp=self.logit_soft_capping_value) for embedding in sparse_embeddings]
                
            loss_mnrl = self.sparse_loss.compute_loss_from_embeddings(sparse_embeddings, labels=None)
            
            loss_mse = 0.0
            if labels is not None and len(sparse_embeddings) >= 3:
                anchors = gather_tensor(sparse_embeddings[0])
                positives = gather_tensor(sparse_embeddings[1])
                negatives = gather_tensor(sparse_embeddings[2])
                labels_gathered = gather_tensor(labels)
                
                pos_scores = self.sparse_loss.similarity_fct(anchors, positives).diagonal() * self.sparse_loss.scale
                neg_scores = self.sparse_loss.similarity_fct(anchors, negatives).diagonal() * self.sparse_loss.scale
                
                student_margin = pos_scores - neg_scores
                
                teacher_margin = labels_gathered.to(student_margin.dtype)
                mean = teacher_margin.mean()
                std = teacher_margin.std()
                if torch.isnan(std) or std == 0.0:
                    std = 1.0
                teacher_margin_norm = (teacher_margin - mean) / (std + 1e-8) * self.teacher_score_scale
                
                loss_mse = torch.nn.functional.mse_loss(student_margin, teacher_margin_norm)
                
            base_loss = self.mnrl_weight * loss_mnrl + self.mse_weight * loss_mse
            losses['sparse_loss'] = base_loss
            
            self.last_mnrl_loss = loss_mnrl.item()
            self.last_mse_loss = loss_mse.item()

        if self.dense_loss is not None:
            dense_loss = self.dense_weight*self.dense_loss.compute_loss_from_embeddings(dense_embeddings, labels)
            losses['dense_loss'] = dense_loss
        
        if self.sparse_loss is not None:
            scale = self._get_scale()
            self.sparse_loss.scale = scale
            self._step += 1
        
        self._reg_step += 1

        return losses


def main():
    parser = argparse.ArgumentParser(description="Train Clustered ColBERT SparseEncoder")
    parser.add_argument("--train_data", type=str, required=True, help="Path to training JSONL file")
    parser.add_argument("--eval_data", type=str, required=True, help="Path to evaluation JSONL file")
    parser.add_argument("--base_model", type=str, default="modernbert-base", help="Base model for MLMTransformer")
    parser.add_argument("--model_type", type=str, default="modern-bert", help="Base model for MLMTransformer")
    parser.add_argument("--output_dir", type=str, default="./outputs_clustered_colbert", help="Directory for saving checkpoints")
    parser.add_argument("--run_name", type=str, default="clustered-colbert-run", help="Run name")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epoch variations")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size per device")
    parser.add_argument("--mini_batch_size", type=int, default=32, help="Mini-batch size for CacheContrastive loss")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--grad_acc", type=int, default=1, help="Gradient accumulation")
    parser.add_argument("--reg_weight", type=float, default=5e-4, help="Regularizer weight")
    parser.add_argument("--activation", type=str, default="log1p_relu", choices=["relu", "log1p_relu"], help="Activation function")
    parser.add_argument("--scale_embeddings", type=eval, default=True, help="Scale embeddings by sqrt(d) for causal models")
    parser.add_argument("--num_workers", type=int, default=8, help="Dataloader worker count")
    parser.add_argument("--aux_weight", type=float, default=0.0, help="Auxiliary weight")
    parser.add_argument("--scale", type=float, default=1.0, help="Sparse loss scale")
    parser.add_argument("--use_wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--mse_weight", type=float, default=1.0, help="Margin MSE loss weight")
    parser.add_argument("--mnrl_weight", type=float, default=1.0, help="MNRL loss weight")
    parser.add_argument("--teacher_score_scale", type=float, default=1.0, help="Teacher score normalization scale")
    
    # Clustered specific args
    parser.add_argument("--num_clusters", type=int, default=8000, help="Number of clusters for lexical compression")
    parser.add_argument("--use_triton", action="store_true", help="Use Triton kernels for clustered operations")
    parser.add_argument("--cluster_update_method", type=str, default="greedy", choices=["greedy", "faiss"], help="Cluster update method: 'greedy' (argmax reassign) or 'faiss' (full KMeans, guarantees no empty clusters)")
    parser.add_argument("--update_freq", type=int, default=5, help="How often (in steps) to re-cluster tokens")
    
    # scale scheduling arguments
    parser.add_argument("--scale_start", type=float, default=1.0, help="Initial scale")
    parser.add_argument("--scale_end", type=float, default=1.0, help="Final scale")
    parser.add_argument("--scale_total_steps", type=int, default=12000, help="Total steps for scale")
    parser.add_argument("--reg_start", type=float, default=0.0, help="Initial FLOPS regularization weight.")
    parser.add_argument("--reg_total_steps", type=int, default=500, help="Total steps for regularizer weight schedule.")
    parser.add_argument("--max_steps", type=int, default=-1, help="Max training steps (-1 = use epochs).")
    parser.add_argument("--unfreeze_embeddings", action="store_true", help="Unfreeze the embedding layer")
    parser.add_argument("--use_fp8", action="store_true", help="Enable FP8 quantization for base weights")
    parser.add_argument("--optim", type=str, default="adamw_torch", help="Optimizer type")
    parser.add_argument("--attn_implementation", type=str, default="sdpa", choices=["sdpa", "flash_attention_2", "eager"], help="Attention implementation to use")
    args, _ = parser.parse_known_args()

    if args.use_wandb and int(os.environ.get("LOCAL_RANK", "0")) == 0:
        print(f"Initializing wandb... (rank {os.environ.get('LOCAL_RANK', '0')})", flush=True)
        wandb.init(project="colbert-hrs", entity="yuansu-university-of-british-columbia", name=args.run_name)

    model_args = {"attn_implementation": args.attn_implementation}
    model_args["device_map"] = {"": int(os.environ.get("LOCAL_RANK", "0"))}

    # Model setup
    print("Initializing ClusteredColbertTransformer...", flush=True)
    mlm_transformer = ClusteredColbertTransformer(
        args.base_model,
        max_seq_length=512,
        model_args=model_args,
        model_type=args.model_type,
        num_clusters=args.num_clusters,
        use_triton=args.use_triton,
        activation=args.activation,
        scale_embeddings=args.scale_embeddings,
        unfreeze_embeddings=args.unfreeze_embeddings,
        use_fp8=args.use_fp8
    )
    print("ClusteredColbertTransformer initialized.", flush=True)
    
    print("Initializing ClusteredColbertEncoder...", flush=True)
    from src.models.ClusteredColbertEncoder import ClusteredColbertEncoder
    model = ClusteredColbertEncoder(
        modules=[mlm_transformer],
        embedding_type="colbert"
    )
    model.similarity_fn_name = "colbert"
    print("ClusteredColbertEncoder initialized.", flush=True)

    if ("bert" not in args.model_type.lower()):
        print("Moving model to bfloat16...", flush=True)
        model.bfloat16()
        print("Model moved to bfloat16.", flush=True)
    
    # Load dataset
    print("Loading datasets...", flush=True)
    try:
        train_dataset = load_dataset("json", data_files=args.train_data, keep_in_memory=False)["train"]
        if "query" in train_dataset.column_names:
            train_dataset = train_dataset.rename_columns({"query": "anchor"})
        if "hard_negative" in train_dataset.column_names:
            train_dataset = train_dataset.rename_columns({"hard_negative": "negative"})
            
        def compute_labels(batch):
            if "score_pos" in batch and "score_hard_neg" in batch:
                return {"label": [p - n for p, n in zip(batch["score_pos"], batch["score_hard_neg"])]}
            elif "score_pos" in batch and "score_easy_neg" in batch:
                return {"label": [p - n for p, n in zip(batch["score_pos"], batch["score_easy_neg"])]}
            else:
                return {"label": [0.0] * len(batch["anchor"])}

        cols_to_remove = [c for c in train_dataset.column_names if c not in ["anchor", "positive", "negative"]]
        train_dataset = train_dataset.map(compute_labels, batched=True, remove_columns=cols_to_remove)
        
        eval_dataset = load_dataset("json", data_files=args.eval_data, keep_in_memory=False)["train"]
        if "query" in eval_dataset.column_names:
            eval_dataset = eval_dataset.rename_columns({"query": "anchor"})
        if "hard_negative" in eval_dataset.column_names:
            eval_dataset = eval_dataset.rename_columns({"hard_negative": "negative"})
            
        MAX_EVAL_SAMPLES = 1000
        if len(eval_dataset) > MAX_EVAL_SAMPLES:
            print(f"Subsampling evaluation dataset from {len(eval_dataset)} to {MAX_EVAL_SAMPLES} to preserve GPU memory.", flush=True)
            eval_dataset = eval_dataset.shuffle(seed=42).select(range(MAX_EVAL_SAMPLES))

        cols_to_remove_eval = [c for c in eval_dataset.column_names if c not in ["anchor", "positive", "negative"]]
        eval_dataset = eval_dataset.map(compute_labels, batched=True, remove_columns=cols_to_remove_eval)

    except Exception as e:
        print(f"Exception during load_dataset: {e}", flush=True)
        raise e
    print("Datasets loaded.", flush=True)

    if "negative" in eval_dataset.column_names:
        evaluator = SparseColBERTTripletEvaluator(
            anchors=eval_dataset["anchor"],
            positives=eval_dataset["positive"],
            negatives=eval_dataset["negative"],
        )
    else:
        print("No negatives found in eval_data, evaluator disabled.", flush=True)
        evaluator = None

    use_grad_cache = args.mini_batch_size < args.batch_size
    loss_class = CachedDistillationLoss if use_grad_cache else DistillationLoss
    
    loss_kwargs = {
        "mse_weight": args.mse_weight,
        "mnrl_weight": args.mnrl_weight,
        "teacher_score_scale": args.teacher_score_scale,
        "model": model,
        "dense_loss": None,
        "sparse_loss": SparseSelfMultipleNegativesRankingLoss(
            model=model, 
            scale=args.scale, 
            similarity_fct=memory_efficient_colbert_similarity
        ),
        "query_regularizer_weight": args.reg_weight,
        "document_regularizer_weight": args.reg_weight,
        "k": None, # Disable top-k for ColBERT full dense outputs
        "aux_weight": args.aux_weight,
        "scale_start": args.scale_start,
        "scale_end": args.scale_end,
        "total_steps": args.scale_total_steps,
        "reg_start": args.reg_start,
        "reg_total_steps": args.reg_total_steps,
    }
    if use_grad_cache:
        loss_kwargs["mini_batch_size"] = args.mini_batch_size

    loss = loss_class(**loss_kwargs)

    training_args = SparseEncoderTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps, 
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_acc,
        learning_rate=args.lr,
        warmup_ratio=0.075,
        weight_decay=1e-4,
        bf16=True,
        gradient_checkpointing=True,
        max_grad_norm=1,
        eval_strategy="steps" if evaluator is not None else "no",
        eval_steps=5750 if evaluator is not None else None,
        save_strategy="epoch",
        save_total_limit=None,
        logging_steps=5,
        dataloader_num_workers=args.num_workers,
        dataloader_drop_last=True,
        lr_scheduler_type='cosine',
        report_to=["wandb"] if args.use_wandb else [],
        run_name=args.run_name,
        optim=args.optim,
        do_train=True,
    )

    class ClusteredTrainer(SparseEncoderTrainer):
        def __init__(self, *args, update_freq=10, cluster_update_method="greedy", **kwargs):
            self._custom_loss_ref = kwargs.get("loss")
            super().__init__(*args, **kwargs)
            self.update_freq = update_freq
            self.step_count = 0
            self.cluster_update_method = cluster_update_method

        def training_step(self, model, inputs, num_items_in_batch=None):
            self.step_count += 1
            if self.step_count % self.update_freq == 0:
                with torch.no_grad():
                    mlm = self.model[0]
                    w = mlm.w
                    num_clusters = mlm.num_clusters
                    use_triton = mlm.use_triton
                    
                    from clustered_splade import get_w_reduced, UnembeddingCompressSparse
                    temp_layer = UnembeddingCompressSparse(num_clusters, use_triton)
                    
                    if self.cluster_update_method == "faiss":
                        new_cluster_ids = temp_layer.update_mask_faiss(w)
                    else:
                        cluster_ids = mlm.cluster_ids
                        w_reduced = get_w_reduced(w, cluster_ids, num_clusters, use_triton)
                        new_cluster_ids = temp_layer.update_mask(w, w_reduced)
                    
                    mlm.cluster_ids.data.copy_(new_cluster_ids)
                    
                    mlm._cached_w_reduced = None
                    mlm._cached_w_version = -1
                    
                    if args.use_wandb and self.state.is_world_process_zero:
                        import wandb
                        wandb.log({"mask_updates": self.step_count // self.update_freq}, step=self.state.global_step)
            
            return super().training_step(model, inputs, num_items_in_batch)
            
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            loss = super().compute_loss(model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch)
            
            # Extract loss components and log to wandb directly
            if self.state.is_world_process_zero and getattr(args, "use_wandb", False):
                # Drill down if model is wrapped in DDP or similar DataParallel layers
                core_model = model
                if hasattr(core_model, "module"):
                    core_model = core_model.module
                    
                loss_obj = self.loss
                if isinstance(loss_obj, dict):
                    loss_obj = list(loss_obj.values())[0] if loss_obj else None
                
                # In SentenceTransformerTrainer, loss might be assigned elsewhere or hidden. 
                # Let's search self.loss, and also the original loss object we passed in kwargs if available
                found_mnrl = False
                target_loss = loss_obj
                
                if hasattr(target_loss, "last_mnrl_loss"):
                    found_mnrl = True
                elif hasattr(target_loss, "loss") and hasattr(target_loss.loss, "last_mnrl_loss"):
                    target_loss = target_loss.loss
                    found_mnrl = True
                    
                if not found_mnrl:
                    # Look for it on the custom ClusteredTrainer instance if someone injected it
                    if hasattr(self, "_custom_loss_ref") and hasattr(self._custom_loss_ref, "last_mnrl_loss"):
                        target_loss = self._custom_loss_ref
                        found_mnrl = True
                
                if found_mnrl and hasattr(target_loss, "last_mnrl_loss") and hasattr(target_loss, "last_mse_loss"):
                    import wandb
                    if wandb.run is not None:
                        wandb.log({
                            "train/mnrl_loss": float(target_loss.last_mnrl_loss),
                            "train/mse_loss": float(target_loss.last_mse_loss),
                        }, step=self.state.global_step, commit=False)
                else:
                    if self.state.global_step % 10 == 0:
                        print(f"DEBUG: Could not find last_mnrl_loss. self.loss type is: {type(loss_obj)}. hasattr(self.loss, 'last_mnrl_loss')={hasattr(loss_obj, 'last_mnrl_loss')}")
                        
            return loss
        def save_model(self, output_dir=None, _internal_call=False):
            super().save_model(output_dir, _internal_call)
            if output_dir is None:
                output_dir = self.args.output_dir
            if output_dir is not None:
                os.makedirs(output_dir, exist_ok=True)
            mlm = self.model[0]
            torch.save(mlm.cluster_ids.cpu(), os.path.join(output_dir, "cluster_ids.pt"))

    trainer = ClusteredTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
        cluster_update_method=args.cluster_update_method, # Re-added these as they are used in ClusteredTrainer __init__
        update_freq=args.update_freq, # Re-added these as they are used in ClusteredTrainer __init__
    )


    if training_args.do_train:
        print(f"Rank {os.environ.get('LOCAL_RANK', '0')}: Starting trainer.train()...", flush=True)
        trainer.train()
        print(f"Rank {os.environ.get('LOCAL_RANK', '0')}: training finished.", flush=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        with open("/tmp/err.log", "w") as f:
            f.write(traceback.format_exc())
            print(f"CRASH OCCURRED: {e}")
        raise
