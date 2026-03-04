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

        from exp2.clustered_splade import UnembeddingCompressSparse
        temp_layer = UnembeddingCompressSparse(num_clusters, use_triton)
        
        model_name_or_path = args[0]
        local_cluster_path = os.path.join(model_name_or_path, "cluster_ids.pt")
        
        if os.path.exists(local_cluster_path):
            print(f"Loading cluster IDs from local checkpoint: {local_cluster_path}")
            cluster_ids_val = torch.load(local_cluster_path, map_location=self.w.device)
        else:
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
                
            from exp2.clustered_splade import get_w_reduced
            current_w_version = getattr(self.w, "_version", 0)
            is_training = torch.is_grad_enabled()
            
            should_recompute = (
                self._cached_w_reduced is None or 
                self._cached_w_version != current_w_version or
                (is_training and getattr(self._cached_w_reduced, 'is_inference', lambda: False)())
            )

            if not should_recompute:
                w_reduced = self._cached_w_reduced
            else:
                w_reduced = get_w_reduced(self.w, self.cluster_ids, self.num_clusters, self.use_triton)
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
    args = parser.parse_args()

    if args.use_wandb:
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
        train_dataset = load_dataset("json", data_files=args.train_data, keep_in_memory=True)["train"]
        if "query" in train_dataset.column_names:
            train_dataset = train_dataset.rename_columns({"query": "anchor"})
        
        eval_dataset = load_dataset("json", data_files=args.eval_data, keep_in_memory=True)["train"]
        
        # Subsample evaluation dataset if too large to avoid OOM during dense tensor accumulation
        MAX_EVAL_SAMPLES = 1000
        if len(eval_dataset) > MAX_EVAL_SAMPLES:
            print(f"Subsampling evaluation dataset from {len(eval_dataset)} to {MAX_EVAL_SAMPLES} to preserve GPU memory.", flush=True)
            eval_dataset = eval_dataset.shuffle(seed=42).select(range(MAX_EVAL_SAMPLES))

        if "query" in eval_dataset.column_names:
            eval_dataset = eval_dataset.rename_columns({"query": "anchor"})
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
    loss_class = CachedSpladeMixedTopKLoss if use_grad_cache else SpladeMixedTopKLoss
    
    loss_kwargs = {
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
        save_only_model=True,
        lr_scheduler_type='cosine',
        report_to=["wandb"] if args.use_wandb else [],
        run_name=args.run_name,
        optim=args.optim,
    )

    class ClusteredTrainer(SparseEncoderTrainer):
        def __init__(self, *args, update_freq=10, cluster_update_method="greedy", **kwargs):
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
                    
                    from exp2.clustered_splade import get_w_reduced, UnembeddingCompressSparse
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
                    
                    if args.use_wandb:
                        import wandb
                        wandb.log({"mask_updates": self.step_count // self.update_freq}, step=self.state.global_step)
            
            return super().training_step(model, inputs, num_items_in_batch)

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
        loss=loss,
        evaluator=evaluator,
        cluster_update_method=args.cluster_update_method,
        update_freq=args.update_freq,
    )

    trainer.train()

if __name__ == "__main__":
    main()
