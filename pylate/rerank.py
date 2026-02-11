import json
import sys
import torch
from pylate import rank
from pylate import evaluation, losses, models, utils

TOP_K = 100


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

        # PyLate output: [{'id': ..., 'score': ...}, ...]
        pred_ids = [
            p["id"] if isinstance(p, dict) else p
            for p in preds
        ]

        for k in ks:
            if any(pid in gt for pid in pred_ids[:k]):
                hits[k] += 1

    return {f"success@{k}": hits[k] / total for k in ks}, total


def main(enriched_path, model):
    # ---- Load data ----
    queries, documents, documents_ids, ground_truth = load_for_reranking(
        enriched_path,
        top_k=TOP_K
    )

    print(f"Loaded {len(queries)} queries for reranking")

    # ---- Encode ----
    queries_embeddings = model.encode(
        queries,
        is_query=True,
    )

    documents_embeddings = model.encode(
        documents,
        is_query=False,
    )

    # ---- Rerank ----
    reranked_documents = rank.rerank(
        documents_ids=documents_ids,
        queries_embeddings=queries_embeddings,
        documents_embeddings=documents_embeddings,
    )
    #print(reranked_documents)
    # ---- Metrics ----
    metrics, n = success_at_k(
        reranked_documents,
        ground_truth,
        ks=(1, 8, 50)
    )

    print(f"\n#queries evaluated: {n}")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python rerank_eval.py <enriched_predictions.jsonl>")
        sys.exit(1)

    enriched_path = sys.argv[1]

    #model_name = "checkpoints/colbert-modernbert-base-maxsim-og/checkpoint-12652"
    #similarity_fn_name = "MaxSim"
    model_name = "checkpoints/colbert-modernbert-base-maskedmean-og-t0.05/checkpoint-12652"
    similarity_fn_name = "MaskedMean"

    model = models.ColBERT(
    model_name_or_path=model_name,
    query_length=512,
    document_length=512,
    similarity_fn_name=similarity_fn_name,
    model_kwargs={"attn_implementation": "sdpa"},
    )

    model = torch.compile(model)
    model.eval()
    with torch.no_grad():
        main(enriched_path, model)
