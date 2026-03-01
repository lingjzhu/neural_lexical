"""
Run cluster_mapping + activation analysis for all models defined in run_eval.sh.
Precomputed cluster files from precomputed_clusters/ are reused whenever possible.

Usage:
    cd /home/slime-base/projects/jian/neural_lexical/exp2
    python analysis/run_all_analysis.py [--models modernbert qwen3 qwen3_diffusion] [--data_path ...]
"""
import os
import sys
import json
import argparse
import torch
import numpy as np
from collections import Counter
from tqdm import tqdm

# Add project root to path
ANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))
EXP2_DIR = os.path.abspath(os.path.join(ANALYSIS_DIR, ".."))
PROJECT_ROOT = os.path.abspath(os.path.join(EXP2_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, EXP2_DIR)


# Model registry (mirrors run_eval.sh)
MODELS = {
    "modernbert": {
        "checkpoint": f"{EXP2_DIR}/outputs_clustered_modernbert_large_4000/checkpoint-1582",
        "base_model": "answerdotai/ModernBERT-large",
        "model_type": "modernbert",
        "num_clusters": 4000,
        "activation": "log1p_relu",
        "precomputed_key": "modernbert_large",
    },
    "qwen3": {
        "checkpoint": f"{EXP2_DIR}/outputs_clustered_qwen3_0.6B_4000_relu/checkpoint-1582",
        "base_model": "Qwen/Qwen3-0.6B",
        "model_type": "qwen3",
        "num_clusters": 4000,
        "activation": "relu",
        "precomputed_key": "qwen_0.6b",
    },
    "qwen3_diffusion": {
        "checkpoint": f"{EXP2_DIR}/outputs_clustered_qwen3_diffusion_0.6B_4000_relu/checkpoint-1582",
        "base_model": "dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1",
        "model_type": "qwen3_diffusion",
        "num_clusters": 4000,
        "activation": "relu",
        "precomputed_key": "qwen3_0.6b_diffusion",
    },
}
PRECOMPUTED_DIR = os.path.join(EXP2_DIR, "precomputed_clusters")
DEFAULT_DATA_PATH = os.path.join(PROJECT_ROOT, "data/amazon_triplets.jsonl")


# ── Cluster ID loading ──────────────────────────────────────────────────────────

def get_cluster_ids(cfg):
    """Load trained cluster IDs from checkpoint (falls back to precomputed)."""
    from transformers import AutoTokenizer

    cluster_path = os.path.join(cfg["checkpoint"], "cluster_ids.pt")
    if os.path.exists(cluster_path):
        print(f"  OK Loading trained cluster_ids from {cluster_path}")
        cluster_ids = torch.load(cluster_path, map_location="cpu").numpy()
        tokenizer = AutoTokenizer.from_pretrained(cfg["checkpoint"])
        return cluster_ids, tokenizer

    k = cfg["num_clusters"]
    precomputed_path = os.path.join(PRECOMPUTED_DIR, f"{cfg['precomputed_key']}_{k}_clusters.pt")
    if os.path.exists(precomputed_path):
        print(f"  OK Loading precomputed cluster_ids from {precomputed_path}")
        cluster_ids = torch.load(precomputed_path, map_location="cpu").numpy()
        tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
        return cluster_ids, tokenizer

    raise FileNotFoundError(
        f"No cluster_ids found for {cfg['model_type']}.\n"
        f"  Tried: {cluster_path}\n"
        f"  Tried: {precomputed_path}"
    )


def get_initial_cluster_ids(cfg):
    """Load initial (pre-training) cluster IDs from precomputed_clusters/."""
    from transformers import AutoTokenizer

    k = cfg["num_clusters"]
    precomputed_path = os.path.join(PRECOMPUTED_DIR, f"{cfg['precomputed_key']}_{k}_clusters.pt")
    if os.path.exists(precomputed_path):
        print(f"  OK Loading initial cluster_ids from {precomputed_path}")
        cluster_ids = torch.load(precomputed_path, map_location="cpu").numpy()
        tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
        return cluster_ids, tokenizer

    raise FileNotFoundError(
        f"No precomputed cluster_ids found for {cfg['model_type']}.\n"
        f"  Tried: {precomputed_path}"
    )


# ── Cluster mapping analysis ────────────────────────────────────────────────────

def _build_and_save_mapping(cluster_ids, tokenizer, num_clusters, out_path, label):
    mapping = {i: [] for i in range(num_clusters)}
    vocab = tokenizer.get_vocab()
    inv_vocab = {v: k for k, v in vocab.items()}
    for token_id, cluster_id in enumerate(cluster_ids):
        if token_id in inv_vocab:
            mapping[int(cluster_id)].append(inv_vocab[token_id])
    sizes = [len(v) for v in mapping.values()]
    print(f"  [{label}] Vocab={len(cluster_ids)}, Clusters={num_clusters}, "
          f"tokens/cluster: min={min(sizes)}, max={max(sizes)}, avg={np.mean(sizes):.1f}")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in mapping.items()}, f, indent=2, ensure_ascii=False)
    print(f"  Saved {label} -> {out_path}")
    return sizes


def run_cluster_mapping(cfg, output_dir):
    print(f"\n[Cluster mapping] {cfg['model_type']}")
    num_clusters = cfg["num_clusters"]

    try:
        cluster_ids_init, tokenizer = get_initial_cluster_ids(cfg)
        init_sizes = _build_and_save_mapping(
            cluster_ids_init, tokenizer, num_clusters,
            os.path.join(output_dir, "cluster_mapping_initial.json"),
            label="initial"
        )
    except FileNotFoundError as e:
        print(f"  WARNING: {e}")
        cluster_ids_init, init_sizes = None, None

    try:
        cluster_ids_trained, tokenizer = get_cluster_ids(cfg)
        trained_sizes = _build_and_save_mapping(
            cluster_ids_trained, tokenizer, num_clusters,
            os.path.join(output_dir, "cluster_mapping_trained.json"),
            label="trained"
        )
    except FileNotFoundError as e:
        print(f"  WARNING: {e}")
        cluster_ids_trained, trained_sizes = None, None

    if init_sizes is not None and trained_sizes is not None:
        init_arr = np.array(init_sizes)
        trained_arr = np.array(trained_sizes)

        if cluster_ids_init is not None and cluster_ids_trained is not None:
            changed = int(np.sum(cluster_ids_init != cluster_ids_trained))
            pct = 100.0 * changed / len(cluster_ids_init)
            print(f"  Token reassignments after training: {changed:,} / {len(cluster_ids_init):,} ({pct:.1f}%)")

        init_empty = int(np.sum(init_arr == 0))
        trained_empty = int(np.sum(trained_arr == 0))
        print(f"  Empty clusters -- initial: {init_empty}, trained: {trained_empty}")

        summary = {
            "initial": {"min": int(init_arr.min()), "max": int(init_arr.max()),
                        "mean": float(init_arr.mean()), "std": float(init_arr.std()),
                        "empty": init_empty},
            "trained": {"min": int(trained_arr.min()), "max": int(trained_arr.max()),
                        "mean": float(trained_arr.mean()), "std": float(trained_arr.std()),
                        "empty": trained_empty},
        }
        if cluster_ids_init is not None and cluster_ids_trained is not None:
            summary["token_reassignment_pct"] = round(pct, 2)
        with open(os.path.join(output_dir, "cluster_distribution_summary.json"), "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print("  Saved distribution summary")


# ── Activation analysis ─────────────────────────────────────────────────────────

def run_activation_analysis(cfg, output_dir, data_path, num_samples=500, batch_size=16):
    print(f"\n[Activation analysis] {cfg['model_type']}")

    if not os.path.exists(data_path):
        print(f"  WARNING: Data file not found: {data_path} -- skipping")
        return

    from evaluate_hrs import ClusteredMLMTransformer, SparseEncoder

    mlm = ClusteredMLMTransformer(
        cfg["checkpoint"],
        max_seq_length=512,
        num_clusters=cfg["num_clusters"],
        use_triton=True,
        activation=cfg["activation"],
        model_type=cfg["model_type"],
    )
    model = SparseEncoder(modules=[mlm])
    mlm.recover_clusters_if_needed(base_model_path=cfg["base_model"], model_dir=cfg["checkpoint"])
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    if "bert" not in cfg["model_type"].lower():
        model.to(torch.bfloat16)

    print(f"  Loading {num_samples} articles from {data_path}...")
    articles = []
    with open(data_path) as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            obj = json.loads(line)
            articles.append(obj.get("positive", ""))

    tokenizer = mlm.tokenizer
    num_clusters = cfg["num_clusters"]

    # Collect full [N x C] activation matrix on CPU (float32)
    all_sparse = []
    all_activations = []

    for i in tqdm(range(0, len(articles), batch_size), desc="  Inference"):
        batch = articles[i : i + batch_size]
        features = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
        features = {k: v.to(model.device) for k, v in features.items()}

        with torch.no_grad():
            with torch.autocast(device_type="cuda", enabled=torch.cuda.is_available(), dtype=torch.bfloat16):
                outputs = model(features)
                sparse = outputs["sparse_embeddings"].float().cpu()  # [B, C]

        all_sparse.append(sparse)

        for idx in range(sparse.shape[0]):
            if i + idx < 10:
                vals, inds = torch.topk(sparse[idx], k=min(50, num_clusters))
                entries = [
                    {"cluster_id": int(ind), "score": float(v)}
                    for v, ind in zip(vals, inds) if v > 0
                ]
                all_activations.append({
                    "article_index": i + idx,
                    "text": batch[idx],
                    "top_activations": entries,
                })

    # Full matrix [N, C]
    mat = torch.cat(all_sparse, dim=0).numpy()  # [N_samples, num_clusters]
    N = mat.shape[0]

    # Per-cluster statistics
    mean = mat.mean(axis=0)          # [C]
    var  = mat.var(axis=0)           # [C]
    std  = mat.std(axis=0)           # [C]
    freq = (mat > 0).mean(axis=0)   # fraction of samples where cluster activates [0, 1]

    # Rank by variance descending
    rank_by_var = np.argsort(var)[::-1]

    cluster_stats = [
        {
            "cluster_id": int(cid),
            "variance": float(var[cid]),
            "std": float(std[cid]),
            "mean": float(mean[cid]),
            # activation_freq: 1.0 = fires in every doc (background), 0.0 = never fires
            "activation_freq": float(freq[cid]),
        }
        for cid in rank_by_var
    ]

    with open(os.path.join(output_dir, "cluster_variance.json"), "w", encoding="utf-8") as f:
        json.dump(cluster_stats, f, indent=2, ensure_ascii=False)

    # frequencies.json (backward compat)
    cluster_counts = {str(i): int((mat[:, i] > 0).sum()) for i in range(num_clusters)}
    with open(os.path.join(output_dir, "frequencies.json"), "w") as f:
        json.dump(cluster_counts, f, indent=2, ensure_ascii=False)
    with open(os.path.join(output_dir, "sample_activations.json"), "w") as f:
        json.dump(all_activations, f, indent=2, ensure_ascii=False)

    n_active = int((freq > 0).sum())
    print(f"  Samples: {N}, Active clusters: {n_active} / {num_clusters}")
    print(f"  Top-10 HIGH-variance clusters (most discriminative):")
    for s in cluster_stats[:10]:
        print(f"    cluster {s['cluster_id']:4d}  var={s['variance']:.4f}  mean={s['mean']:.4f}  freq={s['activation_freq']:.2%}")
    print(f"  Top-10 LOW-variance clusters (background noise):")
    for s in cluster_stats[-10:][::-1]:
        print(f"    cluster {s['cluster_id']:4d}  var={s['variance']:.6f}  mean={s['mean']:.4f}  freq={s['activation_freq']:.2%}")
    print(f"  Saved cluster_variance + frequencies + sample_activations -> {output_dir}")

    del model, mlm
    torch.cuda.empty_cache()


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=list(MODELS.keys()),
                        choices=list(MODELS.keys()), help="Which models to analyse")
    parser.add_argument("--data_path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--skip_activation", action="store_true",
                        help="Skip activation analysis (just do cluster mapping)")
    args = parser.parse_args()

    for model_name in args.models:
        cfg = MODELS[model_name]
        if not os.path.exists(cfg["checkpoint"]):
            print(f"\nWARNING: Checkpoint not found for {model_name}: {cfg['checkpoint']} -- skipping")
            continue

        output_dir = os.path.join(ANALYSIS_DIR, "results", model_name)
        os.makedirs(output_dir, exist_ok=True)
        print(f"\n{'='*60}")
        print(f"  Model: {model_name}")
        print(f"  Output: {output_dir}")
        print(f"{'='*60}")

        try:
            run_cluster_mapping(cfg, output_dir)
        except Exception as e:
            print(f"  ERROR in cluster mapping: {e}")
            import traceback; traceback.print_exc()

        if not args.skip_activation:
            try:
                run_activation_analysis(cfg, output_dir, args.data_path, args.num_samples)
            except Exception as e:
                print(f"  ERROR in activation analysis: {e}")
                import traceback; traceback.print_exc()

    print(f"\nAll done. Results in {ANALYSIS_DIR}/results/")


if __name__ == "__main__":
    main()
