import sys
import os
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
os.environ["HF_TRUST_REMOTE_CODE"] = "1"
os.environ["TRUST_REMOTE_CODE"] = "True"
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
from modeling_t5gemma2 import T5Gemma2TextEncoder, T5Gemma2Config, T5Gemma2TextConfig, patch_t5gemma2

class ClusteredMLMTransformer(MLMTransformer):
    def __init__(self, *args, num_clusters=8000, use_triton=True, activation="relu", model_type="qwen3", scale_embeddings=True, unfreeze_embeddings=False, use_fp8=False, **kwargs):
        if model_type == "llada":
            kwargs["backend"] = "llada"
        elif "qwen3_diffusion" in model_type.lower():
            kwargs["backend"] = "qwen3_diffusion"
        elif "qwen3" in model_type.lower():
            kwargs["backend"] = "qwen3"
        elif model_type == "t5gemma2":
            kwargs["backend"] = "t5gemma2"
        else:
            kwargs["backend"] = "torch"
        super().__init__(*args, **kwargs)
        self.num_clusters = num_clusters
        self.model_type = model_type
        self.scale_embeddings = scale_embeddings
        self.unfreeze_embeddings = unfreeze_embeddings
        self.use_fp8 = use_fp8
        
        # Disable torch.compile which might cause hangs in GradCache hooks
        if hasattr(self.auto_model.config, "reference_compile"):
            self.auto_model.config.reference_compile = False
        
        # We need the original weight
        # Iterate to logically find lm_head or decoder inside arbitrary PEFT or wrapping abstractions
        self.w = None
        for name, module in self.auto_model.named_modules():
            if name.endswith("lm_head") or name.endswith("decoder") or name.endswith("ff_out") or (model_type == "t5gemma2" and name == ""):
                if hasattr(module, "weight") or (model_type == "t5gemma2" and hasattr(module, "embed_tokens")):
                    if model_type == "t5gemma2":
                        self.w = module.embed_tokens.weight
                        break
                    # Check if it's actually the final output layer (e.g. self.transformer.ff_out or self.model.lm_head)
                    if "ff_out" in name and "transformer.ff_out" not in name:
                        continue # there might be other ff_out from internal blocks, skip them if they aren't the final head
                    self.w = module.weight
                    if hasattr(self.w, "modules_to_save"):
                        self.w = self.w.default.weight
                    
                    # Disable original layer by dynamically patching the parent
                    parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
                    if parent_name:
                        parent = self.auto_model.get_submodule(parent_name)
                        setattr(parent, name.split('.')[-1], nn.Identity())
                    else:
                        setattr(self.auto_model, name, nn.Identity())
                    break
                    
        if self.w is None:
            raise ValueError("Could not find lm_head or decoder in auto_model through named_modules")

        self.clustered_layer = ClusteredSpladeFusedMeanPooling(num_clusters, activation=activation, use_triton=use_triton)
        from clustered_splade import UnembeddingCompressSparse
        temp_layer = UnembeddingCompressSparse(num_clusters, use_triton)
        
        # Try to load precomputed clusters
        # Construct filename based on model name and clusters
        model_name_or_path = args[0]
        model_name_or_path = args[0].lower()
        if "qwen3" in model_name_or_path and "diffusion" in model_name_or_path:
            model_key = "qwen3_0.6b_diffusion"
        elif "qwen3-1.7b" in model_name_or_path.lower():
            model_key = "qwen3-1.7b"
        elif "qwen3" in model_name_or_path or "qwen-0.6b" in model_name_or_path:
            model_key = "qwen3-0.6b"
        elif "modernbert-large" in model_name_or_path:
            model_key = "modernbert_large"
        elif "llada" in model_name_or_path:
            model_key = "llada_8b"
        elif "t5gemma-2-1b" in model_name_or_path:
            model_key = "t5gemma2_1b"
        else:
            model_key = model_name_or_path.split('/')[-1].replace('-', '_')
            
        precomputed_path = os.path.join(os.path.dirname(__file__), "precomputed_clusters", f"{model_key}_{num_clusters}_clusters.pt")
        
        if os.path.exists(precomputed_path):
            print(f"Loading precomputed clusters from {precomputed_path}...")
            cluster_ids_val = torch.load(precomputed_path, map_location=self.w.device)
        else:
            print(f"No precomputed clusters found at {precomputed_path}. Initializing with KMeans...")
            cluster_ids_val = temp_layer.init_kmeans(self.w)
            
        self.cluster_ids = nn.Parameter(
            cluster_ids_val, 
            requires_grad=False
        )

    def _load_config(self, model_name_or_path, cache_dir, backend, config_args):
        if backend == "llada":
            # Use local modeling_llada to avoid trust_remote_code; it registers LLaDAConfig
            from src.models.modeling_llada import LLaDAConfig
            from transformers import AutoConfig
            # Register the local config class so AutoConfig can find it
            AutoConfig.register("llada", LLaDAConfig)
            return AutoConfig.from_pretrained(
                model_name_or_path, trust_remote_code=True, cache_dir=cache_dir, **config_args
            ), False
        if backend == "t5gemma2":
            return T5Gemma2Config.from_pretrained(model_name_or_path, cache_dir=cache_dir, **config_args), False
        return super()._load_config(model_name_or_path, cache_dir, backend, config_args)

    def _load_model(self, model_name_or_path, config, backend, is_peft_model, **model_args):
        if backend == "llada":
            from fast_llada import FastLLaDAModel
            
            # Load in bfloat16 and get peft model
            print(f"Loading LLaDA model from {model_name_or_path} with fast_llada (Unsloth)...")
            self.auto_model, self.tokenizer = FastLLaDAModel.from_pretrained(
                model_name=model_name_or_path,
                dtype=torch.bfloat16,
                load_in_4bit=False,
                trust_remote_code=True,
            )
            self.auto_model.gradient_checkpointing = False
            self.auto_model._gradient_checkpointing_func = None
            self.auto_model.supports_gradient_checkpointing = True
            
            # Hugging Face also checks the Class attribute directly in older transformers versions
            self.auto_model.__class__.supports_gradient_checkpointing = True

            self.auto_model = FastLLaDAModel.get_peft_model(
                self.auto_model, 
                r=16, 
                lora_alpha=16, 
                target_modules=["q_proj", "k_proj", "v_proj", "attn_out", "ff_proj", "up_proj", "ff_out"],
                lora_dropout=0, 
                bias="none",
                use_gradient_checkpointing="unsloth",
                random_state=3407,
                use_rslora=False,
            )

            if getattr(self, "unfreeze_embeddings", False):
                print("Unfreezing embedding layer(s) as requested...")
                _unfrozen_count = 0
                for name, param in self.auto_model.named_parameters():
                    # Support multiple naming conventions for embeddings/heads
                    if "wte" in name or "ff_out" in name or "embed_tokens" in name or "lm_head" in name:
                        param.requires_grad = True
                        _unfrozen_count += 1
                print(f"  Unfrozen {_unfrozen_count} embedding/head parameter tensors.")

        elif backend == "t5gemma2":
            print(f"Loading T5Gemma2 text encoder ONLY from {model_name_or_path}...")
            # We only want the text encoder.
            # If model_name_or_path is a full model, we need to load only the encoder.text_model part.
            # Since this is custom modeling, we'll load the full state dict and filter if needed, 
            # or use the fact that they might be stored separately? 
            # Usually from_pretrained handles this if we provide the right class.
            
            # To be safe and follow "remove other components", we instantiate just the text encoder
            self.auto_model = T5Gemma2TextEncoder(config.text_config)
            # Patch with Liger kernels
            patch_t5gemma2(self.auto_model)
            
            # Use transformers internal loading to handle state dict mapping
            from transformers.modeling_utils import load_state_dict
            import os
            
            # Try to find the model file
            model_file = None
            for f in ["pytorch_model.bin", "model.safetensors"]:
                if os.path.exists(os.path.join(model_name_or_path, f)):
                    model_file = os.path.join(model_name_or_path, f)
                    break
            
            if model_file:
                if model_file.endswith(".safetensors"):
                    from safetensors.torch import load_file
                    state_dict = load_file(model_file)
                else:
                    state_dict = torch.load(model_file, map_location="cpu")
                
                # Filter state dict for keys starting with "encoder.text_model."
                new_state_dict = {}
                prefix = "encoder.text_model."
                for k, v in state_dict.items():
                    if k.startswith(prefix):
                        new_state_dict[k[len(prefix):]] = v
                
                if not new_state_dict:
                    # Maybe it's already a text encoder checkpoint?
                    new_state_dict = state_dict
                
                self.auto_model.load_state_dict(new_state_dict, strict=False)
                self.auto_model.to(torch.bfloat16)
                if hasattr(self.auto_model, "gradient_checkpointing_enable"):
                    self.auto_model.gradient_checkpointing_enable()
                del state_dict
                del new_state_dict
            else:
                print(f"Warning: Could not find model weights in {model_name_or_path}. Initializing from scratch.")
            
            # T5Gemma2 doesn't use the standard tokenizer from AutoTokenizer sometimes if it's custom.
            # But the script expects self.tokenizer.
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        else:
            super()._load_model(model_name_or_path, config, backend, is_peft_model, **model_args)

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
            if hasattr(outputs, "logits"):
                hidden_states = outputs.logits
            else:
                hidden_states = outputs.last_hidden_state
            
            # Shape depends on backend. For Qwen3ForEmbedding it's [B, S, H] now. For ModernBert it's [B, S, H]
            if hidden_states.dim() == 2:
                hidden_states = hidden_states.unsqueeze(1) # [B, 1, H]
            
            # Scale embeddings by sqrt(d) for causal models to avoid non-decreasing loss
            if self.scale_embeddings and ("qwen3" in self.model_type.lower() or "llada" in self.model_type.lower() or "t5gemma2" in self.model_type.lower()):
                d = hidden_states.shape[-1]
                hidden_states = hidden_states * (d ** 0.5)
                
            # Apply clustered layer (fused projection + pooling)
            attention_mask = features.get("attention_mask")
            clustered_pooled_logits = self.clustered_layer(hidden_states, self.w, self.cluster_ids, attention_mask)
            
            features["sparse_embeddings"] = clustered_pooled_logits
            
            if self.model_type != "t5gemma2":
                features["dense_embeddings"] = outputs.hidden_states if hasattr(outputs, 'hidden_states') else None
            else:
                features["dense_embeddings"] = outputs.last_hidden_state
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
    parser.add_argument("--scale_embeddings", type=eval, default=True, help="Scale embeddings by sqrt(d) for causal models")
    parser.add_argument("--num_workers", type=int, default=8, help="Dataloader worker count")
    parser.add_argument("--k", type=lambda s: [int(x) for x in s.split(',')], default=None, help="Comma separated list of top-k")
    parser.add_argument("--aux_weight", type=float, default=0.0, help="Auxiliary weight")
    parser.add_argument("--scale", type=float, default=1.0, help="Sparse loss scale")
    parser.add_argument("--use_wandb", action="store_true", help="Enable wandb logging")
    
    # Clustered SPLADE specific args
    parser.add_argument("--num_clusters", type=int, default=8000, help="Number of clusters for lexical compression")
    parser.add_argument("--use_triton", action="store_true", help="Use Triton kernels for clustered operations")
    parser.add_argument("--cluster_update_method", type=str, default="greedy", choices=["greedy", "faiss"], help="Cluster update method: 'greedy' (argmax reassign) or 'faiss' (full KMeans, guarantees no empty clusters)")
    parser.add_argument("--update_freq", type=int, default=5, help="How often (in steps) to re-cluster tokens")
    
    # --- scale scheduling arguments ---
    parser.add_argument("--scale_start", type=float, default=1.0, help="Initial scale")
    parser.add_argument("--scale_end", type=float, default=1.0, help="Final scale")
    parser.add_argument("--scale_total_steps", type=int, default=12000, help="Total steps for scale")
    parser.add_argument("--reg_start", type=float, default=0.0, help="Initial FLOPS regularization weight.")
    parser.add_argument("--reg_total_steps", type=int, default=200, help="Total steps for regularizer weight schedule.")
    parser.add_argument("--max_steps", type=int, default=-1, help="Max training steps (-1 = use epochs).")
    parser.add_argument("--unfreeze_embeddings", action="store_true", help="Unfreeze the embedding layer during LoRA finetuning")
    parser.add_argument("--use_fp8", action="store_true", help="Enable FP8 quantization for base weights")
    parser.add_argument("--optim", type=str, default="adamw_torch", help="Optimizer type (e.g., adamw_torch, adamw_bnb_8bit)")
    parser.add_argument("--attn_implementation", type=str, default="sdpa", choices=["sdpa", "flash_attention_2", "eager"], help="Attention implementation to use")
    args = parser.parse_args()

    if args.use_wandb:
        wandb.init(project="splade-hrs", entity="yuansu-university-of-british-columbia", name=args.run_name)

    model_args = {"attn_implementation": args.attn_implementation}
    if "bert" not in args.model_type.lower():
        model_args["torch_dtype"] = torch.bfloat16

    # Model setup
    mlm_transformer = ClusteredMLMTransformer(
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
        max_steps=args.max_steps,  # -1 = disabled, use epochs
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
        optim=args.optim,
    )

    # Trainer subclass to update mask periodically
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
                    use_triton = mlm.clustered_layer.use_triton
                    
                    from exp2.clustered_splade import get_w_reduced, UnembeddingCompressSparse
                    temp_layer = UnembeddingCompressSparse(num_clusters, use_triton)
                    
                    if self.cluster_update_method == "faiss":
                        new_cluster_ids = temp_layer.update_mask_faiss(w)
                    else:
                        cluster_ids = mlm.cluster_ids
                        w_reduced = get_w_reduced(w, cluster_ids, num_clusters, use_triton)
                        new_cluster_ids = temp_layer.update_mask(w, w_reduced)
                    
                    # Update parameter safely
                    mlm.cluster_ids.data.copy_(new_cluster_ids)
                    
                    # Reset the cache in the fused layer
                    mlm.clustered_layer._cached_w_reduced = None
                    mlm.clustered_layer._cached_w_version = -1
                    
                    if args.use_wandb:
                        import wandb
                        wandb.log({"mask_updates": self.step_count // self.update_freq}, step=self.state.global_step)
            
            return super().training_step(model, inputs, num_items_in_batch)

        def save_model(self, output_dir=None, _internal_call=False):
            super().save_model(output_dir, _internal_call)
            if output_dir is None:
                output_dir = self.args.output_dir
            # Always save the updated cluster_ids alongside the model
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
