import json
import os
import glob
import torch
import wandb
import argparse
from collections import defaultdict
from pylate import rank
from pylate import models
from sentence_transformers.util import batch_to_device
import logging
from pylate.scores import colbert_scores_masked_mean
from pylate.rank import  RerankResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)


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


def parse_conditions(filename):
    hrs = "HRS1" if "HRS1" in filename else "HRS2" if "HRS2" in filename else "UNKNOWN"
    genre = (
        "perGenre" if "perGenre" in filename
        else "crossGenre" if "crossGenre" in filename
        else "UNKNOWN"
    )
    return hrs, genre


def rerank_and_eval(path, model, top_k):
    queries, documents, documents_ids, ground_truth = load_for_reranking(
        path, top_k=top_k
    )
    
    if not queries:
        return None, 0

    if model._similarity_fn_name == "MaxSim":
        logger.info("Using MaxSim!")
        queries_embeddings = model.encode(queries, is_query=True)
        documents_embeddings = model.encode(documents, is_query=False)

        reranked_documents = rank.rerank(
            documents_ids=documents_ids,
            queries_embeddings=queries_embeddings,
            documents_embeddings=documents_embeddings,
        )

    elif model._similarity_fn_name == "MaskedMean":

        logger.info("Using MeanSim!")
        reranked_documents = []

        for query, query_doc_ids, query_docs in zip(
            queries, documents_ids, documents
        ):
            # --- tokenize ---
            q_feat = batch_to_device(model.tokenize([query]), model.device)
            d_feat = batch_to_device(model.tokenize(query_docs), model.device)

            # --- embeddings ---
            with torch.no_grad():
                q_emb = torch.nn.functional.normalize(
                    model(q_feat)["token_embeddings"], p=2, dim=-1
                )  # (1, Tq, K)

                d_emb = torch.nn.functional.normalize(
                    model(d_feat)["token_embeddings"], p=2, dim=-1
                )  # (B_d, Td, K)

            # --- masked mean sim ---
            scores = colbert_scores_masked_mean(
                queries_embeddings=q_emb,
                documents_embeddings=d_emb,
                queries_mask=q_feat["attention_mask"],
                documents_mask=d_feat["attention_mask"],
            )[0]  # (B_d,)

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
            "mrr@20",  # <-- add here
        ]
    )

    aggregated = defaultdict(lambda: defaultdict(list))

    for path in files:
        fname = os.path.basename(path)
        logger.info(f"\n=== Processing {fname} ===")

        metrics, n = rerank_and_eval(path, model, args.top_k)
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
    parser.add_argument("input_dir", help="Directory containing *and_gt.jsonl files")
    parser.add_argument(
        "--model-name",
        default="checkpoints/colbert-modernbert-base-maskedmean-og-t0.05/checkpoint-12652",
        help="ColBERT checkpoint path",
    )
    parser.add_argument(
        "--similarity-fn",
        default="MaskedMean",
        help="Similarity function name",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="Number of candidates per query to rerank",
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

    model = models.ColBERT(
        model_name_or_path=args.model_name,
        query_length=512,
        document_length=512,
        similarity_fn_name=args.similarity_fn,
        model_kwargs={"attn_implementation": "sdpa"},
    )

    model = torch.compile(model)
    model.eval()

    with torch.no_grad():
        main(args, model)

    wandb.finish()
