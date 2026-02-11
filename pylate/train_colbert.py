import argparse
import random
import numpy as np
import torch
from datasets import load_dataset
from sentence_transformers import (
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)

from pylate import evaluation, losses, models, utils
import wandb

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)


def set_seed(seed=233):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_name",
        type=str,
        required=True,
        help="W&B run name and checkpoint directory name",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="W&B run name and checkpoint directory name",
    )
    parser.add_argument(
        "--similarity_fn",
        type=str,
        default="MaskedMean",
        help="Similarity function name for ColBERT (e.g., MaskedMean, Cosine, Dot)",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=1.0,
        help="Similarity function name for ColBERT (e.g., MaskedMean, Cosine, Dot)",
    )
    return parser.parse_args()


args_cli = parse_args()

set_seed()
# =========================
# Config
# =========================
#model_name = "../hiatus/modernbert-base"
batch_size = 64
num_train_epochs = 2

run_name = args_cli.run_name
similarity_fn_name = args_cli.similarity_fn
output_dir = f"checkpoints/{run_name}"

wandb.init(
    project="splade-hrs",
    entity="yuansu-university-of-british-columbia",
    name=run_name,
)

# =========================
# Model
# =========================
model = models.ColBERT(
    model_name_or_path=args_cli.model_name,
    query_length=512,
    document_length=512,
    similarity_fn_name=similarity_fn_name,
    model_kwargs={"attn_implementation": "sdpa"},
)

model = torch.compile(model)
logger.info(f"similarity function is: {model._similarity}")
# =========================
# Data
# =========================

train_dataset = load_dataset(
    "json",
    data_files="../hiatus/training_data_v1_final_dedup.jsonl",
    keep_in_memory=True,
)["train"]

train_dataset = train_dataset.rename_columns({"query":"anchor"})
eval_dataset = load_dataset(
    "json",
    data_files="../hiatus/amazon_triplets.jsonl",
    keep_in_memory=True,
)["train"]


'''
train_dataset = load_dataset("json", data_files="../hiatus/bluesky/train_2epoch_5posts.jsonl", keep_in_memory=True)["train"]#.rename_columns({"query": "anchor"})

eval_dataset = load_dataset("json", data_files="../hiatus/bluesky/20251114_dev_pairs_5post.jsonl", keep_in_memory=True)["train"].select(range(5000))
neg = eval_dataset["positive"].copy()
random.shuffle(neg)

eval_dataset = eval_dataset.add_column("negative", neg)
'''
# =========================
# Loss & Eval
# =========================

#train_loss = losses.Contrastive(model=model, temperature=args_cli.temp, score_metric=model._similarity)
train_loss = losses.CachedContrastive(model=model, mini_batch_size=batch_size, temperature=args_cli.temp)

dev_evaluator = evaluation.ColBERTTripletEvaluator(
    anchors=eval_dataset["anchor"],
    positives=eval_dataset["positive"],
    negatives=eval_dataset["negative"],
)

# =========================
# Training args
# =========================
training_args = SentenceTransformerTrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    gradient_accumulation_steps=1,
    max_grad_norm=1,
    eval_strategy="steps",
    eval_steps=2000,
    save_strategy="epoch",
    fp16=True,
    bf16=False,
    learning_rate=1e-4,
    logging_steps=20,
    dataloader_num_workers=6,
    save_total_limit=5,
    save_only_model=True,
    warmup_steps=1000,
    lr_scheduler_type="cosine",
    gradient_checkpointing=True,
)

# =========================
# Trainer
# =========================
trainer = SentenceTransformerTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=train_loss,
    evaluator=dev_evaluator,
    data_collator=utils.ColBERTSwapCollator(model.tokenize),
)

trainer.train()
