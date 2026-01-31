import argparse
from datasets import load_dataset
from sentence_transformers import SparseEncoderTrainer, SparseEncoderTrainingArguments
from sentence_transformers.sparse_encoder.losses import SparseMultipleNegativesRankingLoss
from sentence_transformers.sparse_encoder.evaluation import SparseInformationRetrievalEvaluator
from collections import defaultdict
import torch
import wandb

from model.pooling import SparseColbertPooling
from model.MLMTransformer import MLMTransformer
from model.SpladeMixedTopKLoss import SpladeColbertTopKLoss
from model.SparseColbertEncoder import SparseColbertEncoder
from model.sparse_colbert_triplet import SparseColBERTTripletEvaluator

def build_ir_evaluator(dataset, name="sparse-ir-eval", limit=5000):
    queries, corpus, relevant_docs = {}, {}, defaultdict(set)
    for i, row in enumerate(dataset):
        qid, did = f"q{i}", f"d{i}"
        queries[qid] = row["anchor"]
        corpus[did] = row["positive"]
        relevant_docs[qid].add(did)
        if i >= limit:
            break
    return SparseInformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name=name,
        accuracy_at_k = [1, 8, 50, 100],
        precision_recall_at_k = [1, 8, 50, 100],
        max_active_dims=2048,
    )

def main():
    parser = argparse.ArgumentParser(description="Train SPLADE SparseEncoder with Sentence Transformers")

    parser.add_argument("--train_data", type=str, required=True, help="Path to training JSONL file")
    parser.add_argument("--eval_data", type=str, required=True, help="Path to evaluation JSONL file")
    parser.add_argument("--base_model", type=str, default="../qwen3-0.6b", help="Base model for MLMTransformer")
    parser.add_argument("--backend", type=str, default="modernbert_sparse_colbert", help="Base model for MLMTransformer")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Directory for saving checkpoints")
    parser.add_argument("--run_name", type=str, default="splade-modernbert-reg-threshold", help="Run name")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epoch variations to pre-generate")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size per device")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--grad_acc", type=int, default=1, help="Gradient acclumuation")
    parser.add_argument("--reg_weight", type=float, default=5e-4, help="Gradient acclumuation")
    parser.add_argument("--activation", type=str, default="log1p_relu", choices=["relu", "log1p_relu"], help="Activation function")
    parser.add_argument("--num_workers", type=int, default=8, help="Dataloader worker count")
    parser.add_argument("--k", type=int, default=128)
    parser.add_argument("--aux_weight", type=float, default=0.0, help="Comma separated list")
    parser.add_argument("--scale", type=float, default=1.0, help="Comma separated list")
    parser.add_argument("--use_wandb", action="store_true", help="Enable wandb logging")

    args = parser.parse_args()

    if args.use_wandb:
        wandb.init(project="colbert-hrs", entity="yuansu-university-of-british-columbia", name=args.run_name)

    # 1️⃣ Model setup
    mlm_transformer = MLMTransformer(
        args.base_model,
        max_seq_length=512,
        model_args={"attn_implementation": "flash_attention_2"},
        backend=args.backend
    )
    mlm_transformer.auto_model.k = args.k
    splade_pooling = SparseColbertPooling(
        pooling_strategy="mean",
        activation_function=args.activation  # NOW CONFIGURABLE
    )
    model = SparseColbertEncoder(
        modules=[mlm_transformer, splade_pooling],
        embedding_type="colbert"  
    )
    #model.bfloat16()
    print(model.similarity_fn_name)
    # 2️⃣ Load datasets
    train_dataset = load_dataset(
        "json",
        data_files=args.train_data,
        keep_in_memory=True,
        )["train"]

    train_dataset = train_dataset.rename_columns({"query":"anchor"})

    eval_dataset = load_dataset(
        "json",
        data_files=args.eval_data,
        keep_in_memory=True,
        )["train"]


    # 3️⃣ Evaluator
    #evaluator = build_ir_evaluator(eval_dataset)
    dev_evaluator = SparseColBERTTripletEvaluator(
        anchors=eval_dataset["anchor"],
        positives=eval_dataset["positive"],
        negatives=eval_dataset["negative"],
    )


    loss = SpladeColbertTopKLoss(
        model=model,
        scale=args.scale
        )

    # 5️⃣ Training arguments
    training_args = SparseEncoderTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=2,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_acc,
        learning_rate=args.lr,
        warmup_ratio=0.15,
        weight_decay=1e-2,
        fp16=True,
        gradient_checkpointing=True,
        max_grad_norm=1,
        eval_strategy="steps",
        eval_steps=20000,
        save_strategy="steps",
        save_steps=2000,
        save_total_limit=1,
        logging_steps=20,
        dataloader_num_workers=args.num_workers,
        save_only_model=True,
        lr_scheduler_type='cosine',
        report_to=["wandb"] if args.use_wandb else [],
        run_name=args.run_name,
    )

    # 6️⃣ Trainer
    trainer = SparseEncoderTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        loss=loss,
        evaluator=dev_evaluator,
    )

    trainer.train()

if __name__ == "__main__":
    main()



#python  train_sparse_colbert.py --train_data bluesky/train_2epoch_5posts.jsonl --eval_data bluesky/20251114_dev_pairs_5post.jsonl --base_model modernbert-base --backend modernbert_sparse_colbert --output_dir checkpoints/modernbert-sparse_colbert --use_wandb --run_name modernbert-sparse_colbert --lr 1e-4 --batch_size 64 --activation relu
    
#python  train_sparse_colbert.py --train_data interpretation/data/interpretation_w1/randomly_selected_train.jsonl --eval_data bluesky/20251114_dev_pairs_5post.jsonl --base_model modernbert-base --backend modernbert_sparse_colbert --output_dir checkpoints/modernbert-sparse_colbert --use_wandb --run_name modernbert-sparse_colbert --lr 1e-5 --batch_size 64 --activation relu