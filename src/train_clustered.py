"""Train SPLADE-Lexicon or SPARCOL-Lexicon on anchor/positive JSONL pairs."""
import argparse
import os

import torch

from datasets import load_dataset
from sentence_transformers import SparseEncoderTrainer, SparseEncoderTrainingArguments
from transformers import TrainerCallback

from src.models.ClusteredTransformer import ClusteredTransformer, clustered_maxsim
from src.models.SparseEncoder import SparseEncoder
from src.loss.SpladeLoss import SpladeMixedTopKLoss
from src.loss.ClusteredLoss import ClusteredRankingLoss


class ClusterUpdateCallback(TrainerCallback):
    def __init__(self, module, loss, update_freq):
        self.module = module
        self.loss = loss
        self.update_freq = update_freq

    def on_step_end(self, args, state, control, **kwargs):
        self.loss._reg_step = state.global_step
        if state.global_step % self.update_freq == 0:
            self.module.update_clusters()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_data", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--base_model", default="answerdotai/ModernBERT-large")
    parser.add_argument("--embedding_type", choices=["splade", "colbert"], default="splade")
    parser.add_argument("--num_clusters", type=int, default=4000)
    parser.add_argument("--activation", choices=["relu", "log1p_relu"], default="relu")
    parser.add_argument("--use_triton", action="store_true")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--reg_weight", type=float, default=5e-3)
    parser.add_argument("--update_freq", type=int, default=10)
    args = parser.parse_args()
    if args.update_freq < 1:
        parser.error("--update_freq must be positive")

    module = ClusteredTransformer(
        args.base_model, num_clusters=args.num_clusters, embedding_type=args.embedding_type,
        activation=args.activation, use_triton=args.use_triton, initialize_clusters=True,
        model_args={"attn_implementation": "sdpa"},
    )
    device = f"cuda:{os.environ.get('LOCAL_RANK', '0')}" if torch.cuda.is_available() else "cpu"
    model = SparseEncoder(modules=[module], device=device)
    dataset = load_dataset("json", data_files=args.train_data, split="train")
    if "query" in dataset.column_names:
        dataset = dataset.rename_column("query", "anchor")
    dataset = dataset.select_columns(["anchor", "positive"])
    similarity_kwargs = {"similarity_fct": clustered_maxsim} if args.embedding_type == "colbert" else {}
    loss = SpladeMixedTopKLoss(
        model=model, dense_loss=None,
        sparse_loss=ClusteredRankingLoss(model, **similarity_kwargs),
        k=None, query_regularizer_weight=args.reg_weight,
        document_regularizer_weight=args.reg_weight, aux_weight=0,
        scale_start=1, scale_end=1, reg_start=0, reg_total_steps=200,
    )
    trainer = SparseEncoderTrainer(
        model=model, train_dataset=dataset, loss=loss,
        args=SparseEncoderTrainingArguments(
            output_dir=args.output_dir, num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size, learning_rate=args.lr,
            warmup_ratio=0.075, weight_decay=1e-4, bf16=True,
            gradient_checkpointing=True, max_grad_norm=1, lr_scheduler_type="cosine",
            save_strategy="epoch", save_only_model=True, report_to=[],
            ddp_find_unused_parameters=False,
        ),
        callbacks=[ClusterUpdateCallback(module, loss, args.update_freq)],
    )
    trainer.train()


if __name__ == "__main__":
    main()
