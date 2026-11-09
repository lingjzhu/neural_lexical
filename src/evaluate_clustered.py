"""Evaluate lexicon retrieval on HRS JSONL, or rerank exported candidates."""
import argparse
from collections import defaultdict
from contextlib import nullcontext
import json
from pathlib import Path
import re

import torch

from src.models.ClusteredTransformer import ClusteredTransformer, clustered_maxsim
from src.models.SparseEncoder import SparseEncoder
from src.sparse_evaluate import build_eval_data


def load_model(path, device):
    module = ClusteredTransformer.load(path, model_kwargs={"attn_implementation": "sdpa"})
    if str(device).startswith("cpu"):
        module.use_triton = False
    return SparseEncoder(modules=[module], device=str(device)).eval()


@torch.inference_mode()
def encode(model, texts):
    features = {k: v.to(model.device) for k, v in model.tokenize(texts).items()}
    context = torch.autocast("cuda", dtype=torch.bfloat16) if model.device.type == "cuda" else nullcontext()
    with context:
        return model(features)["sparse_embeddings"].float()


def prune(embeddings, k):
    if 0 < k < embeddings.shape[-1]:
        values, indices = embeddings.topk(k, dim=-1)
        return torch.zeros_like(embeddings).scatter_(-1, indices, values)
    return embeddings


def ranking_metrics(records):
    """Count retrieval misses, including relevant documents absent from the shortlist."""
    totals = defaultdict(float)
    count = 0
    for record in records:
        relevant = set(record["ground_truth"])
        if not relevant:
            continue
        ids = [p["corpus_id"] for p in record["predictions"]]
        ranks = [i + 1 for i, doc_id in enumerate(ids) if doc_id in relevant]
        count += 1
        for k in (1, 8, 50, 100):
            totals[f"accuracy@{k}"] += float(bool(ranks) and ranks[0] <= k)
            totals[f"recall@{k}"] += len(set(ids[:k]) & relevant) / len(relevant)
        totals["mrr@20"] += 1 / ranks[0] if ranks and ranks[0] <= 20 else 0
    if not count:
        raise ValueError("No queries with ground truth to evaluate")
    return {"num_queries": count, **{k: v / count for k, v in totals.items()}}


def discover_hrs(parent):
    """Match files by dataset prefix, never by position in sorted filename lists."""
    parent = Path(parent)
    groups = defaultdict(dict)
    for role in ("queries", "candidates", "groundtruth"):
        for path in sorted(parent.rglob(f"*{role}*.jsonl")):
            # A full HRS release also contains non-English and non-TA1 tasks.
            if any(part.startswith("HRS") and "_" in part for part in path.parts):
                if not any("_english_" in part for part in path.parts) or "TA1" not in path.parts:
                    continue
            key = re.sub(r"(queries|candidates|groundtruth)", "data", path.name)
            seed = next((part for part in path.parts if part.startswith("seed-")), "")
            group = groups[(seed, key)]
            if role in group:
                raise ValueError(f"Ambiguous {role} files: {group[role]} and {path}")
            group[role] = path
    if not groups:
        raise ValueError(f"No English HRS TA1 JSONL files found under {parent}")
    for key, group in groups.items():
        if set(group) != {"queries", "candidates", "groundtruth"}:
            raise ValueError(f"Incomplete HRS file group {key}: {sorted(group)}")
        yield group["queries"], group["candidates"], group["groundtruth"]


def retrieve(model, queries, corpus, relevant_docs, batch_size=8, corpus_chunk_size=2048,
             top_k=100, k=1024):
    if model[0].embedding_type != "splade":
        raise ValueError("Retrieval requires a SPLADE-Lexicon checkpoint")
    corpus_ids = list(corpus)
    if not corpus_ids:
        raise ValueError("Candidate corpus is empty")
    corpus_embeddings = torch.cat([
        prune(encode(model, [corpus[i] for i in corpus_ids[start:start + batch_size]]), k).cpu()
        for start in range(0, len(corpus_ids), batch_size)])
    query_ids = [i for i in queries if relevant_docs.get(i)]
    for start in range(0, len(query_ids), batch_size):
        ids = query_ids[start:start + batch_size]
        q = prune(encode(model, [queries[i] for i in ids]), k)
        values = torch.empty(len(ids), 0, device=model.device)
        indices = torch.empty(len(ids), 0, device=model.device, dtype=torch.long)
        for offset in range(0, len(corpus_ids), corpus_chunk_size):
            d = corpus_embeddings[offset:offset + corpus_chunk_size].to(model.device)
            scores = q @ d.T
            block_values, block_indices = scores.topk(min(top_k, len(d)), dim=-1)
            combined = torch.cat([values, block_values], dim=-1)
            combined_indices = torch.cat([indices, block_indices + offset], dim=-1)
            values, positions = combined.topk(min(top_k, combined.shape[-1]), dim=-1)
            indices = combined_indices.gather(-1, positions)
        for row, qid in enumerate(ids):
            yield {"query_id": qid, "query": queries[qid], "ground_truth": list(relevant_docs[qid]),
                   "predictions": [{"corpus_id": corpus_ids[index], "text": corpus[corpus_ids[index]],
                                    "score": score}
                                   for index, score in zip(indices[row].tolist(), values[row].tolist())]}


def rerank(model, records, batch_size=8, top_k=100):
    if model[0].embedding_type != "colbert":
        raise ValueError("Reranking requires a SPARCOL-Lexicon checkpoint")
    for record in records:
        predictions = record["predictions"][:top_k]
        if len({p["corpus_id"] for p in predictions}) != len(predictions):
            raise ValueError("Duplicate candidate IDs in reranking input")
        q = encode(model, [record["query"]])
        scored = []
        for start in range(0, len(predictions), batch_size):
            batch = predictions[start:start + batch_size]
            documents = encode(model, [p["text"] for p in batch])
            scores = clustered_maxsim(q, documents).flatten().tolist()
            scored.extend({**p, "score": score} for p, score in zip(batch, scores))
        yield {**record, "predictions": sorted(scored, key=lambda p: p["score"], reverse=True)}


def read_records(path):
    with Path(path).open() as handle:
        yield from (json.loads(line) for line in handle if line.strip())


def evaluate_pairs(model, path, batch_size=8):
    """In-batch pair/triplet validation; separate from HRS retrieval metrics."""
    from src.train_clustered import load_pairs
    dataset = load_pairs(path)
    count, reciprocal, correct, triplet_correct = 0, 0.0, 0, 0
    for start in range(0, len(dataset), batch_size):
        batch = dataset[start:start + batch_size]
        anchors, positives = encode(model, batch["anchor"]), encode(model, batch["positive"])
        similarity = clustered_maxsim if model[0].embedding_type == "colbert" else lambda q, d: q @ d.T
        scores = similarity(anchors, positives)
        labels = torch.arange(len(anchors), device=scores.device)
        ranks = scores.argsort(descending=True).eq(labels[:, None]).nonzero()[:, 1] + 1
        count += len(anchors)
        correct += (ranks == 1).sum().item()
        reciprocal += (1 / ranks.float()).sum().item()
        if "negative" in batch:
            negatives = encode(model, batch["negative"])
            triplet_correct += (scores.diag() > similarity(anchors, negatives).diag()).sum().item()
    if not count:
        raise ValueError("Validation data is empty")
    return {"num_queries": count, "in_batch_accuracy@1": correct / count,
            "in_batch_mrr": reciprocal / count,
            **({"triplet_accuracy": triplet_correct / count} if "negative" in dataset.column_names else {})}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["retrieve", "rerank", "pairs"])
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--output_file", required=True, help="Metrics JSONL, overwritten on each run")
    parser.add_argument("--parent_dir", help="Root of English HRS TA1 JSONL datasets")
    parser.add_argument("--queries")
    parser.add_argument("--candidates")
    parser.add_argument("--groundtruth")
    parser.add_argument("--input", help="Rerank predictions JSONL/directory, or pair validation JSONL")
    parser.add_argument("--preds_dir", help="Write per-query rankings with text and ground truth")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--corpus_chunk_size", type=int, default=2048)
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--k", type=int, default=1024, help="Active SPLADE dimensions; 0 disables pruning")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if min(args.batch_size, args.corpus_chunk_size, args.top_k) < 1 or args.k < 0:
        parser.error("Sizes must be positive and --k nonnegative")
    if args.top_k < 100 and args.mode != "pairs":
        parser.error("--top_k must be >=100 to report accuracy/recall@100")
    if args.mode == "retrieve":
        if args.parent_dir:
            datasets = list(discover_hrs(args.parent_dir))
        elif all([args.queries, args.candidates, args.groundtruth]):
            datasets = [(Path(args.queries), Path(args.candidates), Path(args.groundtruth))]
        else:
            parser.error("Retrieval needs --parent_dir or --queries/--candidates/--groundtruth")
    elif not args.input:
        parser.error("Reranking and pair evaluation need --input")
    model = load_model(args.model_name, args.device)
    results = []
    if args.mode == "pairs":
        results.append({"dataset": args.input, **evaluate_pairs(model, args.input, args.batch_size)})
    else:
        inputs = datasets if args.mode == "retrieve" else (
            sorted(Path(args.input).glob("*.jsonl")) if Path(args.input).is_dir() else [Path(args.input)])
        if not inputs:
            parser.error("No reranking inputs found")
        for item in inputs:
            if args.mode == "retrieve":
                q, c, gt = item
                name = gt.stem
                records = list(retrieve(model, *build_eval_data(q, c, gt), batch_size=args.batch_size,
                                        corpus_chunk_size=args.corpus_chunk_size, top_k=args.top_k, k=args.k))
            else:
                name = item.stem
                records = list(rerank(model, read_records(item), args.batch_size, args.top_k))
            result = {"dataset": name, **ranking_metrics(records)}
            results.append(result)
            if args.preds_dir:
                path = Path(args.preds_dir) / f"{name}.jsonl"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("".join(json.dumps(r) + "\n" for r in records))
            print(json.dumps(result), flush=True)
    # Match the development evaluator's macro averages over dataset/seed files.
    groups = defaultdict(list)
    for result in results:
        name = result["dataset"]
        split = "HRS1" if "HRS1" in name else "HRS2" if "HRS2" in name else None
        genre = "crossGenre" if "crossGenre" in name else "perGenre" if "perGenre" in name else None
        result["record_type"] = "dataset"
        if split and genre and args.mode != "pairs":
            groups[(split, genre)].append(result)
    for (split, genre), group in groups.items():
        keys = [k for k in group[0] if "@" in k]
        results.append({"record_type": "average", "split": split, "genre_type": genre,
                        "num_datasets": len(group), "num_queries": sum(r["num_queries"] for r in group),
                        **{k: sum(r[k] for r in group) / len(group) for k in keys}})
    output = Path(args.output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps({"model": args.model_name, "mode": args.mode, **r}) + "\n"
                              for r in results))


if __name__ == "__main__":
    main()
