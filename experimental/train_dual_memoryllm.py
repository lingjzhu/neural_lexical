import argparse
import sys
import os
import torch
from datasets import load_dataset
from sentence_transformers import SparseEncoderTrainer, SparseEncoderTrainingArguments
from collections import defaultdict
import wandb

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cached_contrastive import CachedContrastive
from transformers import AutoTokenizer

from modeling_dual_memoryllm import DualMemoryLLMConfig, DualMemoryLLMForAuthorshipVerification

from sentence_transformers.model_card import SentenceTransformerModelCardData

class DualMemoryLLMPyLateWrapper(torch.nn.Sequential):
    """
    Wraps the DualMemoryLLMForAuthorshipVerification model so its outputs
    resemble PyLate ColBERT embeddings. This enables compatibility with PyLate's
    CachedContrastive loss, which expects `outputs["token_embeddings"]`
    and properties like `do_query_expansion` and `skiplist`.
    """
    def __init__(self, model):
        super().__init__(model)
        self.module = model
        self.do_query_expansion = False
        self.skiplist = []
        
        self.model_card_data = SentenceTransformerModelCardData()
        self.model_card_data.model = self
        
        # SentenceTransformers evaluator needs this property
        self.similarity_fn_name = "colbert"

    @property
    def device(self):
        return self.module.device

    def forward(self, sentence_feature):
        # We assume dataset inputs are standard: input_ids, attention_mask
        outputs = self.module(
            input_ids=sentence_feature.get("input_ids"),
            attention_mask=sentence_feature.get("attention_mask")
        )
        return {
            "token_embeddings": outputs["authorship_representation"]
        }
        
    def tokenize(self, texts, **kwargs):
        kwargs.pop("task", None) # SentenceTransformers passes task="document", which HF doesn't support
        if hasattr(self, "tokenizer") and self.tokenizer is not None:
            # Tokenize features similar to how PyLate/SentenceTransformers does it natively
            return self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
                **kwargs
            )
        raise ValueError("Tokenizer not set on the DualMemoryLLMPyLateWrapper.")

    def save_pretrained(self, *args, **kwargs):
        return self.module.save_pretrained(*args, **kwargs)

    def bfloat16(self):
        self.module.bfloat16()
        return self

def main():
    parser = argparse.ArgumentParser(description="Train Dual-MemoryLLM using Sentence Transformers and PyLate CachedContrastive")
    parser.add_argument("--train_data", type=str, default="/home/slime-base/projects/jian/neural_lexical/data/training_data_v1_final_dedup.jsonl")
    parser.add_argument("--eval_data", type=str, default="/home/slime-base/projects/jian/neural_lexical/data/amazon_triplets.jsonl")
    parser.add_argument("--base_model", type=str, default="answerdotai/ModernBERT-base", help="Model initialization weights")
    parser.add_argument("--output_dir", type=str, default="./outputs_dual_memoryllm", help="Directory for saving checkpoints")
    parser.add_argument("--run_name", type=str, default="dual-memoryllm-training", help="Run name")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size per device")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=4, help="Dataloader worker count")
    parser.add_argument("--use_wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Max sequence length")
    
    # Dual-Memory Specific
    parser.add_argument("--disable_ngram_memory", action="store_true", help="If set, uses Unigram variant. Else uses N-Gram variant.")
    
    args = parser.parse_args()

    if args.use_wandb:
        wandb.init(project="dual-memoryllm-training", name=args.run_name)

    # 1. Setup Model
    print(f"Loading Base Weights: {args.base_model}...")
    config = DualMemoryLLMConfig.from_pretrained(args.base_model)
    config.use_ngram_memory = not args.disable_ngram_memory
    
    # Specific Engram Configuration
    config.engram_layer_ids = [1, 3] if config.use_ngram_memory else []
    config.engram_vocab_size = [10000, 10000] if config.use_ngram_memory else []
    
    model_underlying = DualMemoryLLMForAuthorshipVerification.from_pretrained(
        args.base_model, 
        config=config, 
        ignore_mismatched_sizes=True
    )
    
    model = DualMemoryLLMPyLateWrapper(model_underlying)

    # We skip tokenizer assignment locally here to rely on SentenceTransformers Trainer
    # However we must handle tokenization in the trainer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model.tokenizer = tokenizer  # Expose tokenizer to SentenceTransformers Trainer

    # 2. Load Dataset
    print(f"Loading Dataset: {args.train_data}...")
    train_dataset = load_dataset(
        "json",
        data_files=args.train_data,
        keep_in_memory=True,
    )["train"]
    
    # Assume dataset has standard format {"anchor": ... "positive": ... "negative": ...}
    # For PyLate Contrastive, we just pass the raw train_dataset if formatted appropriately
    
    # 3. Setup Loss
    print("Initializing PyLate CachedContrastive Loss...")
    loss = CachedContrastive(
        model=model,
        mini_batch_size=max(1, args.batch_size // 4),  # Chunk chunks during forward
        temperature=0.05
    )

    # 4. Training Arguments
    training_args = SparseEncoderTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        weight_decay=1e-4,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        max_grad_norm=1.0,
        save_strategy="epoch",
        logging_steps=5,
        dataloader_num_workers=args.num_workers,
        save_only_model=True,
        lr_scheduler_type='cosine',
        report_to=["wandb"] if args.use_wandb else [],
        run_name=args.run_name,
    )

    # 5. Trainer
    trainer = SparseEncoderTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        loss=loss,
        evaluator=None, # Expand with a generic dev set evaluator if eval_data is passed
    )

    print("Starting Training...")
    trainer.train()

if __name__ == "__main__":
    main()
