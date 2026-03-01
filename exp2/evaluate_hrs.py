import sys
import os
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
os.environ["HF_TRUST_REMOTE_CODE"] = "1"
os.environ["TRUST_REMOTE_CODE"] = "True"
import glob
import json
import torch
import torch.nn as nn
from tqdm import tqdm
from sentence_transformers.sparse_encoder.evaluation import SparseInformationRetrievalEvaluator
from collections import defaultdict

# Add path for local modules
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from src.models.pooling import LightSpladePooling
from src.models.MLMTransformer import MLMTransformer
from src.models.SparseEncoder import SparseEncoder

# Import clustered components
try:
    from fused_clustered_splade import ClusteredSpladeFusedMeanPooling
except ImportError:
    # Handle if imports fail when running from different dirs
    from exp2.fused_clustered_splade import ClusteredSpladeFusedMeanPooling

class ClusteredMLMTransformer(MLMTransformer):
    def __init__(self, *args, num_clusters=8000, use_triton=True, activation="log1p_relu", model_type="modernbert", **kwargs):
        if model_type == "llada":
            kwargs["backend"] = "llada"
        elif "qwen3_diffusion" in model_type.lower():
            kwargs["backend"] = "qwen3_diffusion"
        elif "qwen3" in model_type.lower():
            kwargs["backend"] = "qwen3"
        else:
            kwargs["backend"] = "torch"
        super().__init__(*args, **kwargs)
        self.num_clusters = num_clusters
        self.model_type = model_type
        
        # Disable torch.compile if present
        if hasattr(self.auto_model.config, "reference_compile"):
            self.auto_model.config.reference_compile = False
        
        # Find the unembedding weight (lm_head or decoder or ff_out)
        self.w = None
        for name, module in self.auto_model.named_modules():
            # ModernBERT might use specialized heads
            if name.endswith("lm_head") or name.endswith("decoder") or name.endswith("ff_out") or name.endswith("head"):
                if hasattr(module, "weight"):
                    if "ff_out" in name and "transformer.ff_out" not in name:
                        continue 
                    self.w = module.weight
                    if hasattr(self.w, "modules_to_save"):
                        self.w = self.w.default.weight
                    
                    # Patch the head to Identity to intercept hidden states in logits field
                    parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
                    if parent_name:
                        parent = self.auto_model.get_submodule(parent_name)
                        setattr(parent, name.split('.')[-1], nn.Identity())
                    else:
                        setattr(self.auto_model, name, nn.Identity())
                    print(f"✅ Patched {name} with Identity")
                    break
        
        # Fallback to tok_embeddings if no output head found (tied weights)
        if self.w is None:
            print("⚠️ Could not find output head, falling back to tok_embeddings")
            for name, module in self.auto_model.named_modules():
                if "tok_embeddings" in name or "word_embeddings" in name:
                    if hasattr(module, "weight"):
                        self.w = module.weight
                        if hasattr(self.w, "modules_to_save"):
                            self.w = self.w.default.weight
                        break

        if self.w is None:
            raise ValueError("Could not find lm_head or word_embeddings in auto_model")

        self.clustered_layer = ClusteredSpladeFusedMeanPooling(num_clusters, activation=activation, use_triton=use_triton)
        
        # Determine vocab size from weight
        vocab_size = self.w.shape[0]
        print(f"🔹 Intercepted weight with vocab_size={vocab_size}")

        # Register cluster_ids as a parameter
        self.cluster_ids = nn.Parameter(torch.zeros(vocab_size, dtype=torch.long), requires_grad=False)

    def _load_model(self, model_name_or_path, config, backend, is_peft_model, **model_args):
        if backend == "llada":
            from fast_llada import FastLLaDAModel
            
            # Load in fp8 and get peft model
            self.auto_model, self.tokenizer = FastLLaDAModel.from_pretrained(
                model_name=model_name_or_path,
                dtype=torch.bfloat16,
                load_in_4bit=False,
                trust_remote_code=True,
            )

            # Apply manual FP8 casting bypass
            for name, module in self.auto_model.named_modules():
                if "ff_out" in name:
                    continue
                if hasattr(module, "base_layer") and hasattr(module.base_layer, "weight"):
                    W = module.base_layer.weight.data
                    if W.dtype in (torch.float16, torch.bfloat16, torch.float32):
                        scale = W.abs().max() / 448.0
                        scale = scale.to(torch.float32)
                        W_fp8 = (W / scale).to(torch.float8_e4m3fn)
                        module.base_layer.weight.data = W_fp8
                        module.base_layer.weight_scale = scale.view(1)
        else:
            super()._load_model(model_name_or_path, config, backend, is_peft_model, **model_args)

    def recover_clusters_if_needed(self, base_model_path=None, model_dir=None):
        """Recover cluster_ids via K-Means if they are all zeros, or load them if they exist."""
        if torch.all(self.cluster_ids == 0):
            import os
            
            # 1. Try to load from checkpoint directly
            if model_dir and os.path.exists(os.path.join(model_dir, "cluster_ids.pt")):
                cluster_path = os.path.join(model_dir, "cluster_ids.pt")
                print(f"🔹 Found trained cluster_ids at {cluster_path}. Loading...")
                loaded_ids = torch.load(cluster_path, map_location=self.cluster_ids.device)
                self.cluster_ids.data.copy_(loaded_ids)
                print("✅ Cluster recovery from checkpoint complete.")
                return

            # 2. Otherwise trigger K-Means fall-back from untrained weights
            print(f"⚠️ cluster_ids are all zeros. Triggering deterministic recovery via K-Means from {base_model_path}...")
            if not base_model_path:
                raise ValueError("base_model_path must be provided to recover cluster_ids!")
            from transformers import AutoModelForMaskedLM, AutoConfig
            
            if self.model_type == "llada":
                from fast_llada import FastLLaDAModel
                base_auto_model, _ = FastLLaDAModel.from_pretrained(base_model_path, dtype=torch.bfloat16, load_in_4bit=False, trust_remote_code=True)
            else:
                config = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)
                base_auto_model = AutoModelForMaskedLM.from_pretrained(base_model_path, config=config, trust_remote_code=True)
                
            base_w = None
            for name, module in base_auto_model.named_modules():
                if name.endswith("lm_head") or name.endswith("decoder") or name.endswith("ff_out") or name.endswith("head"):
                    if hasattr(module, "weight"):
                        if "ff_out" in name and "transformer.ff_out" not in name:
                            continue 
                        base_w = module.weight
                        if hasattr(base_w, "modules_to_save"):
                            base_w = base_w.default.weight
                        break
            if base_w is None:
                for name, module in base_auto_model.named_modules():
                    if "tok_embeddings" in name or "word_embeddings" in name:
                        if hasattr(module, "weight"):
                            base_w = module.weight
                            if hasattr(base_w, "modules_to_save"):
                                base_w = base_w.default.weight
                            break
            
            from exp2.clustered_splade import UnembeddingCompressSparse
            temp_layer = UnembeddingCompressSparse(self.num_clusters, self.clustered_layer.use_triton)
            # Use deterministic random_state to match training
            recovered_ids = temp_layer.init_kmeans(base_w)
            self.cluster_ids.data.copy_(recovered_ids)
            del base_auto_model
            print("✅ Cluster recovery complete.")

    def forward(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        trans_features = {
            key: value
            for key, value in features.items()
            if key in ["input_ids", "attention_mask", "token_type_ids"]
        }

        # auto_model with Identity head returns hidden states in 'logits'
        outputs = self.auto_model(**trans_features)
        
        try:
            hidden_states = outputs.logits
            # Handle different output shapes (Model-specific)
            if hidden_states.dim() == 2:
                hidden_states = hidden_states.unsqueeze(1)
                
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


def load_predictions(pred_path):
    preds = []
    with open(pred_path, "r", encoding="utf-8") as f:
        for line in f:
            preds.append(json.loads(line))
    return preds


def enrich_predictions(predictions, corpus, relevant_docs, top_k=None):
    enriched = []

    for pred in predictions:
        qid = pred["query_id"]
        query_text = pred["query"]

        gt_docs = []
        for cid in relevant_docs.get(qid, []):
            gt_docs.append(cid)

        results = pred["results"][:top_k] if top_k else pred["results"]
        pred_docs = []
        for r in results:
            cid = r["corpus_id"]
            pred_docs.append({
                "corpus_id": cid,
                "score": r["score"],
                "text": corpus.get(cid)
            })

        enriched.append({
            "query_id": qid,
            "query": query_text,
            "ground_truth": gt_docs,
            "predictions": pred_docs
        })

    return enriched

def save_enriched(enriched, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for record in enriched:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def find_jsonl_files(parent_dir):
    ground_truth_files = glob.glob(os.path.join(parent_dir, "**/*groundtruth*.jsonl"), recursive=True)
    query_files = glob.glob(os.path.join(parent_dir, "**/*queries*.jsonl"), recursive=True)
    candidate_files = glob.glob(os.path.join(parent_dir, "**/*candidates*.jsonl"), recursive=True)

    if not ground_truth_files or not query_files or not candidate_files:
        raise RuntimeError(f"No valid eval sets found in {parent_dir}")

    ground_truth_files.sort()
    query_files.sort()
    candidate_files.sort()

    print(f"Found {len(ground_truth_files)} ground_truth, {len(query_files)} queries, {len(candidate_files)} candidates")
    return ground_truth_files, query_files, candidate_files


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_eval_data(queries_path, candidates_path, ground_truth_path):
    queries_raw = load_jsonl(queries_path)
    candidates_raw = load_jsonl(candidates_path)
    ground_truth_raw = load_jsonl(ground_truth_path)

    doc_to_authors = {
        item["documentID"]: item.get("authorIDs", [])
        for item in ground_truth_raw
        if "documentID" in item
    }

    queries = {q["documentID"]: q["fullText"].strip() for q in queries_raw if q.get("fullText", "").strip()}
    corpus = {c["documentID"]: c["fullText"].strip() for c in candidates_raw if c.get("fullText", "").strip()}

    relevant_docs = {}
    for qid in queries:
        q_authors = set(doc_to_authors.get(qid, []))
        if not q_authors:
            continue
        relevant_docs[qid] = [
            cid for cid, c_text in corpus.items()
            if q_authors.intersection(doc_to_authors.get(cid, []))
        ]

    relevant_docs = {k: v for k, v in relevant_docs.items() if v}

    return queries, corpus, relevant_docs

def find_ta1_dirs(root_dir):
    ta1_dirs = {"HRS1": [], "HRS2": []}

    for path in glob.glob(os.path.join(root_dir, "**/TA1"), recursive=True):
        norm = os.path.normpath(path)
        if "HRS1_english_long" in norm:
            ta1_dirs["HRS1"].append(path)
        elif "HRS2_english_medium" in norm:
            ta1_dirs["HRS2"].append(path)

    if not ta1_dirs["HRS1"] and not ta1_dirs["HRS2"]:
        raise RuntimeError(f"No TA1 folders found under {root_dir}")

    return ta1_dirs

def find_prediction_file(output_dir, set_name):
    base = set_name.rsplit("_", 1)[0]
    base = base.replace(".jsonl", "")

    pattern = os.path.join(
        output_dir,
        f"*{base}*predictions_dot.jsonl"
    )

    matches = glob.glob(pattern)

    if not matches:
        raise FileNotFoundError(f"No prediction file found for base={base} in {output_dir}")

    return matches[0]

def extract_metrics(metrics):
    """
    Extract accuracy@100, accuracy@8, and mrr@20 safely.
    """
    out = {}
    for k, v in metrics.items():
        if not isinstance(v, (int, float)):
            continue
        if k.endswith("accuracy@100"):
            out["accuracy@100"] = v
        elif k.endswith("accuracy@8"):
            out["accuracy@8"] = v
        elif k.endswith("mrr@20"):
            out["mrr@20"] = v
    return out


def average_metrics(metrics_list):
    avg = {}
    if not metrics_list:
        return avg
    keys = metrics_list[0].keys()
    for k in keys:
        avg[k] = sum(m[k] for m in metrics_list) / len(metrics_list)
    return avg

def get_genre_type(dataset_name: str):
    if "crossGenre" in dataset_name:
        return "crossGenre"
    if "perGenre" in dataset_name:
        return "perGenre"
    return "unknown"


def main(parent_dataset_dir, model_name, output_file, args):
    ta1_dirs = find_ta1_dirs(parent_dataset_dir)

    if args.clustered:
        mlm_transformer = ClusteredMLMTransformer(
            model_name,
            max_seq_length=512,
            model_args={"attn_implementation": "sdpa"},
            num_clusters=args.num_clusters,
            use_triton=args.use_triton,
            activation=args.activation,
            model_type=args.model_type
        )
        model = SparseEncoder(
            modules=[mlm_transformer],
            prompts={"query": " ", "passage": " "}
        )
        mlm_transformer.recover_clusters_if_needed(base_model_path=args.base_model, model_dir=model_name)
    else:
        mlm_transformer = MLMTransformer(
            model_name,
            max_seq_length=512,
            model_args={"attn_implementation": "sdpa"},
            backend=args.backend
        )
        splade_pooling = LightSpladePooling(
            pooling_strategy="mean",
            activation_function="log1p_relu"
        )
        model = SparseEncoder(
            modules=[mlm_transformer, splade_pooling],
            prompts={"query": " ", "passage": " "}
        )
    
    model.eval()
    if "bert" not in args.model_type.lower():
        model.to(torch.bfloat16)

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    summary = defaultdict(list) # key: (split, genre_type)

    print("\n🔹 Starting evaluation")

    for split, dirs in ta1_dirs.items():
        for ta1_dir in dirs:
            ground_truth_files, query_files, candidate_files = find_jsonl_files(ta1_dir)

            for i, (q_path, c_path, g_path) in enumerate(
                zip(query_files, candidate_files, ground_truth_files)
            ):
                # Extract seed name (e.g., seed-0) from the path
                seed_name = "unknown_seed"
                path_parts = g_path.split(os.sep)
                for part in path_parts:
                    if part.startswith("seed-"):
                        seed_name = part
                        break

                dataset_base = os.path.basename(g_path).replace(".jsonl", "")
                model_base_name = os.path.basename(os.path.normpath(model_name))
                if model_base_name.startswith("checkpoint"):
                    parent_dir_name = os.path.basename(os.path.dirname(os.path.normpath(model_name)))
                    model_base_name = f"{parent_dir_name}_{model_base_name}"
                set_name = f"{model_base_name}_{seed_name}_{split}_{dataset_base}_{i}"
                genre_type = get_genre_type(dataset_base)
                
                if genre_type is None:
                    continue

                print(f"\n🔸 Evaluating {set_name} (genre={genre_type}, path={g_path})")

                queries, corpus, relevant_docs = build_eval_data(
                    q_path, c_path, g_path
                )

                if not queries or not corpus or not relevant_docs:
                    continue

                evaluator = SparseInformationRetrievalEvaluator(
                    queries=queries,
                    corpus=corpus,
                    relevant_docs=relevant_docs,
                    name=set_name,
                    accuracy_at_k=[1, 8, 50, 100],
                    mrr_at_k=[20],
                    max_active_dims=args.k,
                    show_progress_bar=True,
                    batch_size=64,
                    write_predictions=True
                )

                with torch.autocast(device_type="cuda", enabled=torch.cuda.is_available(), dtype=torch.bfloat16):
                    with torch.no_grad():
                        metrics = evaluator(model, output_path=args.preds_dir if not args.disable_preds else None)

                norm = extract_metrics(metrics)
                if not norm:
                    continue
                summary[(split, genre_type)].append(norm)

                metrics_out = {
                    "record_type": "eval",
                    "split": split,
                    "dataset": set_name,
                    "model": model_base_name,
                    "genre_type": genre_type,
                    **metrics,
                }

                with open(output_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(metrics_out) + "\n")

                if not args.disable_preds:
                    try:
                        pred_file = find_prediction_file(args.preds_dir, set_name)
                        predictions = load_predictions(pred_file)
                        enriched = enrich_predictions(
                            predictions=predictions,
                            corpus=corpus,
                            relevant_docs=relevant_docs,
                            top_k=100
                        )
                        out_path = os.path.join(
                            args.preds_dir,
                            f"{set_name}_predictions_with_text_and_gt.jsonl"
                        )
                        save_enriched(enriched, out_path)
                    except Exception as e:
                        print(f"⚠️ Failed to save enriched predictions for {set_name}: {e}")

    # Final averages
    with open(output_file, "a", encoding="utf-8") as f:
        for (split, genre_type), values in summary.items():
            avg = average_metrics(values)
            avg_out = {
                "record_type": "average",
                "split": split,
                "genre_type": genre_type,
                **avg,
            }
            f.write(json.dumps(avg_out) + "\n")

            print(f"\n📊 Final average for {split} / {genre_type}")
            for k, v in avg.items():
                print(f"   {k}: {v:.4f}")

    print(f"\n✅ All results saved to {output_file}")


def print_final_results(path):
    grouped = defaultdict(list)

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            rec = json.loads(line)
            if rec.get("record_type") != "eval": continue
            
            split = rec.get("split")
            dataset = rec.get("dataset")
            
            if split is None or dataset is None:
                continue
                
            genre_type = get_genre_type(dataset) or rec.get("genre_type")
            if genre_type is None:
                continue
            
            metrics = extract_metrics(rec)
            if not metrics:
                continue

            grouped[(split, genre_type)].append(metrics)

    if not grouped:
        print("❌ No valid evaluation records found")
        return

    print("\n📊 Averages from existing results\n")
    for (split, genre), values in grouped.items():
        avg = {}
        keys = values[0].keys()
        for k in keys:
            avg[k] = sum(v[k] for v in values) / len(values)
            
        print(f"{split} / {genre}")
        for k, v in avg.items():
            print(f"  {k}: {v:.6f}")
            
        print(json.dumps({
            "record_type": "average",
            "split": split,
            "genre_type": genre,
            **avg
        }))
        print()
        
        with open(path, "a", encoding="utf-8") as out:
            avg_out = {
                "record_type": "average",
                "split": split,
                "genre_type": genre,
                **avg
            }
            out.write(json.dumps(avg_out) + "\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate HRS with Clustered or Standard SPLADE")
    parser.add_argument("--parent_dir", required=True, type=str, help="Parent directory containing evaluation sets")
    parser.add_argument("--model_name", required=True, type=str, help="Model path")
    parser.add_argument("--base_model", default="answerdotai/ModernBERT-large", type=str, help="Base model for recovering original cluster_ids")
    parser.add_argument("--backend", default="modernbert_fused_mean", type=str)
    parser.add_argument("--output_file", default="sparse_hrs_evaluation.jsonl", type=str)
    parser.add_argument("--preds_dir", default="hrs_preds", type=str)
    parser.add_argument("--k", default=8000, type=int)
    parser.add_argument("--model_type", default="modernbert", type=str)
    parser.add_argument("--clustered", action="store_true", help="Use ClusteredMLMTransformer")
    parser.add_argument("--num_clusters", default=8000, type=int)
    parser.add_argument("--use_triton", action="store_true")
    parser.add_argument("--activation", default="log1p_relu", type=str)
    parser.add_argument("--disable_preds", action="store_true", help="Disable saving predictions and enriched predictions")

    args = parser.parse_args()
    if not os.path.exists(args.preds_dir):
        os.makedirs(args.preds_dir, exist_ok=True)
        
    model_base_name = os.path.basename(os.path.normpath(args.model_name))
    if model_base_name.startswith("checkpoint"):
        parent_dir_name = os.path.basename(os.path.dirname(os.path.normpath(args.model_name)))
        model_base_name = f"{parent_dir_name}_{model_base_name}"
        
    if args.output_file == "sparse_hrs_evaluation.jsonl":
        args.output_file = f"eval_{model_base_name}.jsonl"
    
    main(args.parent_dir, args.model_name, args.output_file, args)
    print_final_results(args.output_file)

