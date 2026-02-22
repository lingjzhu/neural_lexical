import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
from datasets import load_dataset
from sentence_transformers import SparseEncoderTrainer, SparseEncoderTrainingArguments
from sentence_transformers.sparse_encoder.evaluation import SparseInformationRetrievalEvaluator

from collections import defaultdict
import torch
import torch.nn as nn
import wandb

from src.models.pooling import LightSpladePooling
from src.models.MLMTransformer import MLMTransformer
from src.loss.SpladeLoss import SpladeMixedTopKLoss, SparseSelfMultipleNegativesRankingLoss, CachedSpladeMixedTopKLoss
from src.models.SparseEncoder import SparseEncoder
from fused_clustered_splade import ClusteredSpladeFusedMeanPooling

class ClusteredMLMTransformer(MLMTransformer):
    def __init__(self, *args, num_clusters=8000, use_triton=True, activation="relu", model_type="qwen3", **kwargs):
        kwargs["backend"] = "torch"
        super().__init__(*args, **kwargs)
        self.num_clusters = num_clusters
        self.model_type = model_type
        
        # Disable torch.compile which might cause hangs in GradCache hooks
        if hasattr(self.auto_model.config, "reference_compile"):
            self.auto_model.config.reference_compile = False
        
        # We need the original weight
        # Assumes the backend has a linear layer named 'lm_head' or 'decoder' etc.
        if hasattr(self.auto_model, 'lm_head'):
            self.w = self.auto_model.lm_head.weight
            self.auto_model.lm_head = nn.Identity() # disable original
        elif hasattr(self.auto_model, 'decoder'):
            self.w = self.auto_model.decoder.weight
            self.auto_model.decoder = nn.Identity()
        else:
            raise ValueError("Could not find lm_head or decoder in auto_model")

        self.clustered_layer = ClusteredSpladeFusedMeanPooling(num_clusters, activation=activation, use_triton=use_triton)
        from clustered_splade import UnembeddingCompressSparse
        temp_layer = UnembeddingCompressSparse(num_clusters, use_triton)
        self.cluster_ids = nn.Parameter(
            temp_layer.init_kmeans(self.w), 
            requires_grad=False
        )

    def forward(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        trans_features = {
            key: value
            for key, value in features.items()
            if key in ["input_ids", "attention_mask", "token_type_ids"]
        }

        # Instead of self.auto_model(**trans_features), we do it manually to intercept hidden states
        # but auto_model with Identity lm_head will return hidden states in the logits field!
        outputs = self.auto_model(**trans_features)
        
        try:
            hidden_states = outputs.logits
            # Shape depends on backend. For Qwen3ForEmbedding it's [B, H]. For ModernBert it's [B, S, H]
            if hidden_states.dim() == 2:
                hidden_states = hidden_states.unsqueeze(1) # [B, 1, H]
                
            # Apply clustered layer (fused projection + pooling)
            attention_mask = features.get("attention_mask")
            clustered_pooled_logits = self.clustered_layer(hidden_states, self.w, self.cluster_ids, attention_mask)
            
            features["sparse_embeddings"] = clustered_pooled_logits
            
            if self.model_type != "t5gemma":
                features["dense_embeddings"] = outputs.hidden_states if hasattr(outputs, 'hidden_states') else None
            else:
                features["dense_embeddings"] = outputs.decoder_hidden_states if hasattr(outputs, 'decoder_hidden_states') else None
        except AttributeError:
            features = None

        return features

    def get_sentence_embedding_dimension(self) -> int:
        return self.num_clusters

def build_ir_evaluator(dataset, name="sparse-ir-eval", limit=5000, k=None):
    queries, corpus, relevant_docs = {}, {}, defaultdict(set)
    for i, row in enumerate(dataset):
        qid, did = f"q{i}", f"d{i}"
        queries[qid] = row["anchor"]
        corpus[did] = row["positive"]
        relevant_docs[qid].add(did)
        if i >= limit:
            break
    max_act_dim = k if k is not None else 8000
    return SparseInformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name=name,
        accuracy_at_k = [1, 8, 50, 100],
        precision_recall_at_k = [1, 8, 50, 100],
        max_active_dims=max_act_dim,
    )

def main():
    parser = argparse.ArgumentParser(description="Train Clustered SPLADE SparseEncoder")
    parser.add_argument("--train_data", type=str, required=True, help="Path to training JSONL file")
    parser.add_argument("--eval_data", type=str, required=True, help="Path to evaluation JSONL file")
    parser.add_argument("--base_model", type=str, default="../qwen3-0.6b", help="Base model for MLMTransformer")
    parser.add_argument("--model_type", type=str, default="qwen3", help="Base model for MLMTransformer")
    parser.add_argument("--output_dir", type=str, default="./outputs_clustered", help="Directory for saving checkpoints")
    parser.add_argument("--run_name", type=str, default="clustered-splade-run", help="Run name")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epoch variations")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size per device")
    parser.add_argument("--mini_batch_size", type=int, default=32, help="Mini-batch size for CacheContrastive loss")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--grad_acc", type=int, default=1, help="Gradient accumulation")
    parser.add_argument("--reg_weight", type=float, default=5e-4, help="Regularizer weight")
    parser.add_argument("--activation", type=str, default="log1p_relu", choices=["relu", "log1p_relu"], help="Activation function")
    parser.add_argument("--num_workers", type=int, default=8, help="Dataloader worker count")
    parser.add_argument("--k", type=lambda s: [int(x) for x in s.split(',')], default=None, help="Comma separated list of top-k")
    parser.add_argument("--aux_weight", type=float, default=0.0, help="Auxiliary weight")
    parser.add_argument("--scale", type=float, default=1.0, help="Sparse loss scale")
    parser.add_argument("--use_wandb", action="store_true", help="Enable wandb logging")
    
    # Clustered SPLADE specific args
    parser.add_argument("--num_clusters", type=int, default=8000, help="Number of clusters for lexical compression")
    parser.add_argument("--use_triton", action="store_true", help="Use Triton kernels for clustered operations")
    
    # --- scale scheduling arguments ---
    parser.add_argument("--scale_start", type=float, default=1.0, help="Initial scale")
    parser.add_argument("--scale_end", type=float, default=1.0, help="Final scale")
    parser.add_argument("--scale_total_steps", type=int, default=12000, help="Total steps for scale")
    parser.add_argument("--reg_start", type=float, default=0.0, help="Initial FLOPS regularization weight.")
    parser.add_argument("--reg_total_steps", type=int, default=200, help="Total steps for regularizer weight schedule.")

    args = parser.parse_args()

    if args.use_wandb:
        wandb.init(project="splade-hrs", entity="yuansu-university-of-british-columbia", name=args.run_name)

    # Model setup
    mlm_transformer = ClusteredMLMTransformer(
        args.base_model,
        max_seq_length=512,
        model_args={"attn_implementation": "sdpa"},
        model_type=args.model_type,
        num_clusters=args.num_clusters,
        use_triton=args.use_triton,
        activation=args.activation
    )
    
    # No separate pooling needed as it is fused in the transformer module
    model = SparseEncoder(
        modules=[mlm_transformer],
        prompts={"query": " ", "passage": " "}  
    )

    if "bert" not in args.model_type.lower():
        model.bfloat16()
    
    print(model.similarity_fn_name)
    
    # Load dataset
    train_dataset = load_dataset("json", data_files=args.train_data, keep_in_memory=True)["train"]
    if "query" in train_dataset.column_names:
        train_dataset = train_dataset.rename_columns({"query": "anchor"})
    eval_dataset = load_dataset("json", data_files=args.eval_data, keep_in_memory=True)["train"]

    max_act_dim = args.k[-1] if args.k else None
    evaluator = build_ir_evaluator(eval_dataset, k=max_act_dim)

    use_grad_cache = args.mini_batch_size < args.batch_size
    loss_class = CachedSpladeMixedTopKLoss if use_grad_cache else SpladeMixedTopKLoss
    
    loss_kwargs = {
        "model": model,
        "dense_loss": None,
        "sparse_loss": SparseSelfMultipleNegativesRankingLoss(model=model, scale=args.scale),
        "query_regularizer_weight": args.reg_weight,
        "document_regularizer_weight": args.reg_weight,
        "k": args.k, 
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
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_acc,
        learning_rate=args.lr,
        warmup_ratio=0.075,
        weight_decay=1e-4,
        bf16=True,
        gradient_checkpointing=True,
        max_grad_norm=1,
        eval_strategy="steps",
        eval_steps=750,
        save_strategy="epoch",
        save_steps=750,
        save_total_limit=1,
        logging_steps=5,
        dataloader_num_workers=args.num_workers,
        save_only_model=True,
        lr_scheduler_type='cosine',
        report_to=["wandb"] if args.use_wandb else [],
        run_name=args.run_name,
    )

    # Trainer subclass to update mask periodically
    class ClusteredTrainer(SparseEncoderTrainer):
        def training_step(self, model, inputs, num_items_in_batch=None):
            # Optional: update mask every N steps
            # Since KMeans provides good init, we just let embeddings train
            return super().training_step(model, inputs, num_items_in_batch)

    trainer = ClusteredTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        loss=loss,
        evaluator=evaluator,
    )

    trainer.train()

if __name__ == "__main__":
    main()
