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
sys.modules['flash_attn.bert_padding'] = m
sys.modules['flash_attn.layers'] = m
sys.modules['flash_attn.layers.rotary'] = m
sys.modules['flash_attn.ops'] = m
sys.modules['flash_attn.ops.triton'] = m
sys.modules['flash_attn.ops.triton.rotary'] = m
sys.modules['flash_attn_2_cuda'] = m

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import glob
import torch
import wandb
import argparse
from collections import defaultdict
from typing_extensions import TypedDict
from sentence_transformers.util import batch_to_device
import logging


from src.models.ClusteredColbertEncoder import ClusteredColbertEncoder
from exp2.train_clustered_colbert import ClusteredColbertTransformer, memory_efficient_colbert_similarity

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)



class RerankResult(TypedDict):
    """
    Rerank result for ranking.

    Parameters
    ----------
    id
        The document id.
    score
        The document score.
    """

    id: int | str
    score: float



def load_for_reranking(enriched_path, top_k=100):
    queries = []
    documents = []
    documents_ids = []
    ground_truth = []

    with open(enriched_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)

            preds = item.get("predictions", [])[:top_k]
            gt = set(item.get("ground_truth", []))

            if not preds or not gt:
                continue

            queries.append(item["query"])
            documents.append([p["text"] for p in preds])
            documents_ids.append([p["corpus_id"] for p in preds])
            ground_truth.append(gt)

    return queries, documents, documents_ids, ground_truth


def success_at_k(reranked_docs, ground_truth, ks=(1, 8, 50)):
    hits = {k: 0 for k in ks}
    total = 0

    for preds, gt in zip(reranked_docs, ground_truth):
        if not gt:
            continue

        total += 1
        pred_ids = [p["id"] if isinstance(p, dict) else p for p in preds]

        for k in ks:
            if any(pid in gt for pid in pred_ids[:k]):
                hits[k] += 1

    return {f"success@{k}": hits[k] / total for k in ks}, total

# Add this function near the top with your other metric functions
def mrr_at_k(reranked_docs, ground_truth, k=20):
    rr_sum = 0.0
    total = 0

    for preds, gt in zip(reranked_docs, ground_truth):
        if not gt:
            continue

        total += 1
        pred_ids = [p["id"] if isinstance(p, dict) else p for p in preds]

        rr = 0.0
        for rank_idx, pid in enumerate(pred_ids, start=1):
            if pid in gt:
                rr = 1.0 / rank_idx
                break

        rr_sum += rr

    return rr_sum / total if total > 0 else 0.0


def apply_topk_mask(x, k):
    if k is None or k <= 0 or k >= x.shape[-1]:
        return x
    
    # x shape: [B, S, C]
    values, indices = torch.topk(x, k, dim=-1)
    
    # Create a zero tensor and scatter the top values
    new_x = torch.zeros_like(x)
    new_x.scatter_(-1, indices, values)
    return new_x


def parse_conditions(filename):
    hrs = "HRS1" if "HRS1" in filename else "HRS2" if "HRS2" in filename else "UNKNOWN"
    genre = (
        "perGenre" if "perGenre" in filename
        else "crossGenre" if "crossGenre" in filename
        else "UNKNOWN"
    )
    return hrs, genre


def rerank_and_eval(path, model, top_k, top_k_vec=None):
    queries, documents, documents_ids, ground_truth = load_for_reranking(
        path, top_k=top_k
    )
    
    if not queries:
        return None, 0
    device = "cuda"

    reranked_documents = []

    for query, query_doc_ids, query_docs in zip(
        queries, documents_ids, documents
    ):  
        doc_batch_size = 32
        device = model.device
        scores_all = []
        # --- tokenize ---
        q_feat = batch_to_device(
            model.tokenize([query]), device
                )

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            # Clustered model returns [B, S, C] dense tensor
            q_embeds = model(q_feat)["sparse_embeddings"]
            
            if top_k_vec is not None:
                q_embeds = apply_topk_mask(q_embeds, top_k_vec)

        # ------------------
        # Encode docs in batches
        # ------------------
        for i in range(0, len(query_docs), doc_batch_size):
            docs_batch = query_docs[i : i + doc_batch_size]

            d_feat = batch_to_device(
                model.tokenize(docs_batch), device
            )

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                d_embeds = model(d_feat)["sparse_embeddings"]
                
                if top_k_vec is not None:
                    d_embeds = apply_topk_mask(d_embeds, top_k_vec)

            # Use memory efficient similarity for dense [B, S, C] tensors
            batch_scores = memory_efficient_colbert_similarity(
                q=q_embeds,
                d=d_embeds,
            )[0]  # Q = 1 → shape [B]

            scores_all.append(batch_scores)

            # optional but helps fragmentation
            del d_feat, d_embeds
            torch.cuda.empty_cache()

        scores = torch.cat(scores_all, dim=0)
        #scores = torch_chunked_topk_sum_sim(qvals, qinds,dvals, dinds, top_n=1)[0]  

        # --- sort ---
        sorted_scores, sorted_indices = torch.sort(scores, descending=True)
        sorted_scores = sorted_scores.cpu().tolist()

        sorted_doc_ids = [query_doc_ids[i] for i in sorted_indices.tolist()]

        reranked_documents.append(
            [
                RerankResult(id=doc_id, score=score)
                for doc_id, score in zip(sorted_doc_ids, sorted_scores)
            ]
        )


    metrics, n = success_at_k(
        reranked_documents,
        ground_truth,
        ks=(1, 8, 50),
    )
    
    mrr20 = mrr_at_k(reranked_documents, ground_truth, k=20)
    metrics["mrr@20"] = mrr20

    return metrics, n


def main(args, model):
    files = sorted(glob.glob(os.path.join(args.input_dir, "*and_gt.jsonl")))
    if not files:
        logger.info("No *and_gt.jsonl files found.")
        return

    per_file_table = wandb.Table(
        columns=[
            "file",
            "HRS",
            "Genre",
            "n_queries",
            "success@1",
            "success@8",
            "success@50",
            "mrr@20",  
        ]
    )

    aggregated = defaultdict(lambda: defaultdict(list))

    for path in files:
        fname = os.path.basename(path)
        logger.info(f"\n=== Processing {fname} ===")

        metrics, n = rerank_and_eval(path, model, args.top_k, top_k_vec=args.top_k_vec)
        if metrics is None:
            logger.info("Skipped (no valid queries)")
            continue

        hrs, genre = parse_conditions(fname)

        per_file_table.add_data(
            fname,
            hrs,
            genre,
            n,
            metrics["success@1"],
            metrics["success@8"],
            metrics["success@50"],
            metrics["mrr@20"],  # <-- add here
        )

        for k, v in metrics.items():
            aggregated[(hrs, genre)][k].append(v)

        logger.info(f"#queries: {n}")
        for k, v in metrics.items():
            logger.info(f"{k}: {v:.4f}")

    wandb.log({"per_file_results": per_file_table})

    for (hrs, genre), metric_dict in aggregated.items():
        wandb.log({
            f"{hrs}/{genre}/{k}": sum(vals) / len(vals)
            for k, vals in metric_dict.items()
        })

    logger.info("\n=== Aggregated Results ===")
    for (hrs, genre), metric_dict in aggregated.items():
        logger.info(f"\n[{hrs} / {genre}]")
        for k, vals in metric_dict.items():
            logger.info(f"{k}: {sum(vals) / len(vals):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("ColBERT reranking evaluation (W&B synced)")
    parser.add_argument("input_dir", default="/home/slimelab/Projects/neural_lexical/exp2/hrs_preds/hrs_preds/modernbert-fused-mean-reg5e-3-scale1-1024", help="Directory containing *and_gt.jsonl files")
    parser.add_argument(
        "--model-name",
        default="outputs_clustered_colbert_modernbert_large_2000_relu/checkpoint-1750", # Example path
        help="ColBERT checkpoint path",
    )
    parser.add_argument(
        "--num_clusters",
        type=int,
        default=2000,
        help="Number of clusters",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="modern-bert-large",
        help="Model type (modern-bert or modern-bert-large)",
    )
    parser.add_argument(
        "--similarity-fn",
        default="MaxSim",
        help="Similarity function name",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="Number of candidates per query to rerank",
    )    
    parser.add_argument(
        "--top_k_vec",
        type=int,
        default=None,
        help="Top-k sparsity for each vector in the sequence (e.g. 128, 512)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=32,
        help="DEPRECATED: Number of candidates per query to rerank",
    )
    parser.add_argument(
        "--wandb-project",
        default="reranking-eval",
        help="W&B project name",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional W&B run name (defaults to input directory name)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="modernbert_sparse_colbert",
        help="Optional W&B run name (defaults to input directory name)",
    )
    parser.add_argument(
        "--act",
        type=str,
        default="log1p_relu",
        help="Optional W&B run name (defaults to input directory name)",
    )
    parser.add_argument(
        "--scale-embeddings",
        type=eval,
        default=True,
        help="Scale embeddings by sqrt(d) for causal models",
    )

    args = parser.parse_args()

    run_name = args.run_name or os.path.basename(os.path.normpath(args.input_dir))

    wandb.init(
        project=args.wandb_project,
        name=run_name,
        entity="yuansu-university-of-british-columbia",
        config={
            "top_k": args.top_k,
            "model": args.model_name,
            "similarity_fn": args.similarity_fn,
            "input_dir": args.input_dir,
        },
    )

    # Extract model type from checkpoint path or default
    mlm_transformer = ClusteredColbertTransformer(
        args.model_name,
        max_seq_length=512,
        model_args={"attn_implementation": "sdpa"},
        model_type=args.model_type,
        num_clusters=args.num_clusters,
        activation=args.act.replace("log1p_", "") if "log1p" in args.act else args.act,
        scale_embeddings=args.scale_embeddings
    )

    model = ClusteredColbertEncoder(
        modules=[mlm_transformer],
        embedding_type="colbert"  
    )
    model.eval()
    if "modernbert" not in args.model_name.lower():
        model.bfloat16()

    with torch.no_grad():
        main(args, model)

    wandb.finish()
