import os
import glob
import json
import torch
from tqdm import tqdm
from src.models.SparseEncoder import SparseEncoder
from src.models.MLMTransformer import MLMTransformer
from src.models.pooling import SpladePooling
from sentence_transformers.sparse_encoder.evaluation import SparseInformationRetrievalEvaluator

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

    # Map documentID → authorIDs
    doc_to_authors = {
        item["documentID"]: item.get("authorIDs", [])
        for item in ground_truth_raw
        if "documentID" in item
    }

    # Prepare texts
    queries = {q["documentID"]: q["fullText"].strip() for q in queries_raw if q.get("fullText", "").strip()}
    corpus = {c["documentID"]: c["fullText"].strip() for c in candidates_raw if c.get("fullText", "").strip()}

    # Build relevance mapping: query_docID → list of candidate_docIDs sharing any authorID
    relevant_docs = {}
    for qid in queries:
        q_authors = set(doc_to_authors.get(qid, []))
        if not q_authors:
            continue
        relevant_docs[qid] = [
            cid for cid, c_text in corpus.items()
            if q_authors.intersection(doc_to_authors.get(cid, []))
        ]

    # Remove queries without any relevant docs
    relevant_docs = {k: v for k, v in relevant_docs.items() if v}

    return queries, corpus, relevant_docs


def main(parent_dir, model_name, output_file):
    ground_truth_files, query_files, candidate_files = find_jsonl_files(parent_dir)
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mlm_transformer = MLMTransformer(
        model_name,
        max_seq_length=350,
        model_args={"attn_implementation": "flash_attention_2"}
    )
    splade_pooling = SpladePooling(pooling_strategy="mean", activation_function="log1p_relu")
    model = SparseEncoder(modules=[mlm_transformer, splade_pooling])
    model.bfloat16()
    print(model.max_seq_length)

    results = []

    print("\n🔹 Starting evaluation ...")

    for i, (q_path, c_path, g_path) in enumerate(zip(query_files, candidate_files, ground_truth_files)):
        set_name = os.path.basename(g_path).strip("_-") or f"set_{i}"
        print(f"\n🔸 Evaluating {set_name}")

        queries, corpus, relevant_docs = build_eval_data(q_path, c_path, g_path)
        n_q, n_c, n_rel = len(queries), len(corpus), sum(len(v) for v in relevant_docs.values())
        print(f"   #queries={n_q}, #candidates={n_c}, #relevant_links={n_rel}")

        if n_q == 0 or n_c == 0 or n_rel == 0:
            print(f"⚠️  Skipping {set_name}: incomplete or empty data")
            continue

        evaluator = SparseInformationRetrievalEvaluator(
            queries=queries,
            corpus=corpus,
            relevant_docs=relevant_docs,
            name=set_name,
            accuracy_at_k = [1, 8, 50, 100],
        )

        try:
            with torch.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
                metrics = evaluator(model, output_path=None)
        except RuntimeError as e:
            print(f"❌ Runtime error in {set_name}: {e}")
            continue

        metrics["dataset"] = set_name
        results.append(metrics)

        with open(output_file, "a", encoding="utf-8") as fout:
            fout.write(json.dumps(metrics) + "\n")

    print(f"\n✅ Evaluation finished. Results saved to {output_file}")
    print(f"Total evaluated sets: {len(results)}")



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate dense retrieval with SentenceTransformers")
    parser.add_argument("--parent_dir", default="hrs_release_11-22-24/hrs_release_11-22-24/HRS_evaluation_samples/HRS2_english_medium/TA1", type=str, help="Parent directory containing evaluation sets")
    parser.add_argument("--model_name", default="splade-modernbert-base-crossgenre_v1/splade-modernbert-base-crossgenre_v1/checkpoint-56768", type=str, help="Model name or path for SentenceTransformer")
    parser.add_argument("--output_file", default="sparse_evaluation.jsonl", type=str, help="Path to save JSONL results")

    args = parser.parse_args()
    main(args.parent_dir, args.model_name, args.output_file)

