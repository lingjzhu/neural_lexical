"""Offline CLI/gradient tests: python -m src.test_clustered_workflow."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import torch

from src.test_clustered import create_fixture
from src.models.ClusteredTransformer import ClusteredTransformer
from src.models.SparseEncoder import SparseEncoder
from src.loss.ClusteredLoss import ClusteredTrainingLoss, CachedClusteredTrainingLoss
from src.evaluate_clustered import discover_hrs, ranking_metrics, read_records, load_model, encode, prune


def run(*args):
    subprocess.run([sys.executable, "-m", *map(str, args)], check=True,
                   env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "USE_TF": "0"})


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


def main():
    torch.manual_seed(0)
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        base = root / "base"
        base.mkdir()
        create_fixture(base)
        rows = [{"anchor": "A sample", "positive": "A longer text", "negative": "text"},
                {"anchor": "longer text", "positive": "sample text", "negative": "A"},
                {"anchor": "sample", "positive": "A", "negative": "longer"},
                {"anchor": "A text", "positive": "text sample", "negative": "sample"}]
        data = root / "pairs.jsonl"
        write_jsonl(data, rows)
        for mode in ("splade", "colbert"):
            module = ClusteredTransformer(str(base), num_clusters=3, embedding_type=mode, max_seq_length=16)
            model = SparseEncoder(modules=[module], device="cpu").eval()
            features = [model.tokenize([r[key] for r in rows]) for key in ("anchor", "positive", "negative")]
            direct = ClusteredTrainingLoss(model, reg_warmup_steps=0)
            cached = CachedClusteredTrainingLoss(model, mini_batch_size=2, reg_warmup_steps=0)
            direct.step = cached.objective.step = 1
            loss = direct(features)
            loss.backward()
            gradients = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
            model.zero_grad()
            cached_loss = cached(features)
            cached_loss.backward()
            torch.testing.assert_close(loss, cached_loss)
            for name, parameter in model.named_parameters():
                if name in gradients:
                    torch.testing.assert_close(parameter.grad, gradients[name], atol=1e-6, rtol=0.001)
            out = root / mode
            command = ["src.train_clustered", "--train_data", data, "--eval_data", data,
                       "--output_dir", out, "--base_model", base, "--initial_clusters", base / "cluster_ids.pt",
                       "--embedding_type", mode, "--num_clusters", "3", "--batch_size", "2",
                       "--mini_batch_size", "1", "--eval_batch_size", "2", "--max_seq_length", "16",
                       "--save_steps", "1", "--eval_steps", "1", "--update_freq", "1", "--device", "cpu"]
            run(*command, "--max_steps", "1")
            run(*command, "--max_steps", "2", "--resume_from_checkpoint", "latest")
            state = json.loads((out / "checkpoint-2" / "trainer_state.json").read_text())
            assert state["global_step"] == 2 and (out / "checkpoint-2" / "optimizer.pt").exists()
            run("src.evaluate_clustered", "pairs", "--model_name", out / "final", "--input", data,
                "--output_file", root / f"{mode}-pairs.jsonl", "--batch_size", "2", "--device", "cpu")

        hrs = root / "seed-0" / "HRS2_english_medium" / "TA1"
        queries = [{"documentID": "q0", "fullText": "A sample"},
                   {"documentID": "q1", "fullText": "longer text"}]
        corpus = [{"documentID": f"d{i}", "fullText": text}
                  for i, text in enumerate(["sample text", "A", "text", "longer"])]
        truth = [{"documentID": row["documentID"], "authorIDs": [str(i % 2)]}
                 for i, row in enumerate(queries + corpus)]
        for role, records in (("queries", queries), ("candidates", corpus), ("groundtruth", truth)):
            write_jsonl(hrs / role / f"HRS2_english_medium_sample-0_TA1_{role}.jsonl", records)
        assert len(list(discover_hrs(root))) == 1
        predictions = root / "retrieved"
        run("src.evaluate_clustered", "retrieve", "--model_name", root / "splade/final",
            "--parent_dir", root, "--output_file", root / "retrieval-metrics.jsonl",
            "--preds_dir", predictions, "--batch_size", "2", "--corpus_chunk_size", "1", "--device", "cpu")
        records = list(read_records(next(predictions.glob("*.jsonl"))))
        model = load_model(str(root / "splade/final"), "cpu")
        scores = prune(encode(model, [q["fullText"] for q in queries]), 1024) @ prune(
            encode(model, [d["fullText"] for d in corpus]), 1024).T
        for row, record in enumerate(records):
            for candidate in record["predictions"]:
                index = int(candidate["corpus_id"][1:])
                assert abs(candidate["score"] - scores[row, index].item()) < 1e-5
        run("src.evaluate_clustered", "rerank", "--model_name", root / "colbert/final",
            "--input", predictions, "--output_file", root / "rerank-metrics.jsonl",
            "--preds_dir", root / "reranked", "--batch_size", "2", "--device", "cpu")
        # MRR@20 cuts off at rank 20; absent relevant documents still count as misses.
        metrics = ranking_metrics([
            {"ground_truth": ["hit"], "predictions": [{"corpus_id": str(i)} for i in range(20)] + [{"corpus_id": "hit"}]},
            {"ground_truth": ["absent"], "predictions": []}])
        assert metrics["num_queries"] == 2 and metrics["mrr@20"] == 0 and metrics["accuracy@100"] == 0.5
        (hrs / "candidates" / "HRS2_english_medium_sample-0_TA1_candidates.jsonl").unlink()
        try:
            list(discover_hrs(root))
        except ValueError:
            pass
        else:
            raise AssertionError("Incomplete HRS groups must fail")
    print("Cached gradients, train/eval/resume, HRS retrieval/reranking and metric edge cases passed")


if __name__ == "__main__":
    main()
