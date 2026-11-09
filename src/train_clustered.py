"""Train and validate SPLADE-Lexicon or SPARCOL-Lexicon from JSONL text pairs."""
import argparse
import os
from pathlib import Path

import torch
from datasets import load_dataset
from sentence_transformers import SparseEncoderTrainer, SparseEncoderTrainingArguments
from transformers import TrainerCallback
from transformers.trainer_utils import get_last_checkpoint

from src.models.ClusteredTransformer import ClusteredTransformer
from src.models.SparseEncoder import SparseEncoder
from src.loss.ClusteredLoss import ClusteredTrainingLoss, CachedClusteredTrainingLoss


class ClusterUpdateCallback(TrainerCallback):
    def __init__(self, module, loss, update_freq):
        self.module, self.loss, self.update_freq = module, loss, update_freq

    def on_step_begin(self, args, state, control, **kwargs):
        objective = getattr(self.loss, "objective", self.loss)
        objective.step = state.global_step

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.update_freq == 0:
            self.module.update_clusters()


class ClusteredTrainer(SparseEncoderTrainer):
    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        # The MLM weights and learned assignments use the module's portable format.
        restored = ClusteredTransformer.load(str(resume_from_checkpoint))
        self.model[0].auto_model.load_state_dict(restored.auto_model.state_dict())
        self.model[0].cluster_ids.copy_(restored.cluster_ids.to(self.model[0].cluster_ids.device))


def load_pairs(path):
    dataset = load_dataset("json", data_files=path, split="train")
    if "query" in dataset.column_names and "anchor" not in dataset.column_names:
        dataset = dataset.rename_column("query", "anchor")
    return dataset.select_columns(["anchor", "positive"] +
                                  (["negative"] if "negative" in dataset.column_names else []))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_data", required=True)
    parser.add_argument("--eval_data", help="Validation JSONL pairs or triplets")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--base_model", default="answerdotai/ModernBERT-large")
    parser.add_argument("--model_name", help="Initialize from a released lexicon checkpoint")
    parser.add_argument("--resume_from_checkpoint", help="Checkpoint directory or 'latest'; restores optimizer/RNG/step")
    parser.add_argument("--embedding_type", choices=["splade", "colbert"], default="splade")
    parser.add_argument("--num_clusters", type=int, default=4000)
    parser.add_argument("--activation", choices=["relu", "log1p_relu"], default="relu")
    parser.add_argument("--use_triton", action="store_true")
    parser.add_argument("--initial_clusters", help="Precomputed cluster_ids.pt for a new run")
    parser.add_argument("--epochs", type=float, default=2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--mini_batch_size", type=int, default=0, help="Enable gradient caching when smaller than batch_size")
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--grad_acc", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--scale", type=float, default=1)
    parser.add_argument("--reg_weight", type=float, help="Defaults: SPLADE 5e-3; SPARCOL 0")
    parser.add_argument("--reg_warmup_steps", type=int, default=200)
    parser.add_argument("--update_freq", type=int, default=10)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", help="Defaults to local CUDA rank, or CPU")
    args = parser.parse_args()
    if min(args.update_freq, args.batch_size, args.eval_batch_size, args.grad_acc,
           args.max_seq_length, args.eval_steps, args.save_steps) < 1 or args.mini_batch_size < 0:
        parser.error("Batch sizes, frequencies, length and accumulation must be positive")
    device = args.device or (f"cuda:{os.environ.get('LOCAL_RANK', '0')}" if torch.cuda.is_available() else "cpu")
    if device.startswith("cuda"):
        torch.cuda.set_device(device)
    resume = args.resume_from_checkpoint
    if resume == "latest":
        resume = get_last_checkpoint(args.output_dir) if Path(args.output_dir).exists() else None
        if resume is None:
            parser.error("No checkpoint found in --output_dir")
    if resume and not (Path(resume) / "optimizer.pt").exists():
        parser.error("Resume needs a training checkpoint with optimizer state; use --model_name for weights-only checkpoints")
    if resume or args.model_name:
        module = ClusteredTransformer.load(resume or args.model_name,
                                          model_kwargs={"attn_implementation": "sdpa"})
        module.max_seq_length = args.max_seq_length
    else:
        module = ClusteredTransformer(
            args.base_model, num_clusters=args.num_clusters, embedding_type=args.embedding_type,
            activation=args.activation, use_triton=args.use_triton,
            initialize_clusters=not args.initial_clusters, cluster_ids_path=args.initial_clusters,
            max_seq_length=args.max_seq_length, model_args={"attn_implementation": "sdpa"})
    if device == "cpu":
        module.use_triton = False
    model = SparseEncoder(modules=[module], device=device)
    reg = args.reg_weight if args.reg_weight is not None else (0.005 if module.embedding_type == "splade" else 0)
    cache = 0 < args.mini_batch_size < args.batch_size
    loss_class = CachedClusteredTrainingLoss if cache else ClusteredTrainingLoss
    loss = loss_class(model, scale=args.scale, reg_weight=reg, reg_warmup_steps=args.reg_warmup_steps,
                      **({"mini_batch_size": args.mini_batch_size} if cache else {}))
    trainer = ClusteredTrainer(
        model=model, train_dataset=load_pairs(args.train_data),
        eval_dataset=load_pairs(args.eval_data) if args.eval_data else None, loss=loss,
        args=SparseEncoderTrainingArguments(
            output_dir=args.output_dir, num_train_epochs=args.epochs, max_steps=args.max_steps,
            per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.eval_batch_size,
            gradient_accumulation_steps=args.grad_acc, learning_rate=args.lr,
            warmup_ratio=0.075, weight_decay=1e-4, bf16=device.startswith("cuda"),
            use_cpu=device == "cpu", gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False}, max_grad_norm=1,
            lr_scheduler_type="cosine", eval_strategy="steps" if args.eval_data else "no",
            eval_steps=args.eval_steps, save_strategy="steps", save_steps=args.save_steps,
            save_only_model=False, report_to=[], seed=args.seed, dataloader_drop_last=int(os.environ.get("WORLD_SIZE", "1")) > 1,
            ddp_find_unused_parameters=False,
        ), callbacks=[ClusterUpdateCallback(module, loss, args.update_freq)],
    )
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(str(Path(args.output_dir) / "final"))
    if args.eval_data:
        trainer.save_metrics("eval", trainer.evaluate())


if __name__ == "__main__":
    main()
