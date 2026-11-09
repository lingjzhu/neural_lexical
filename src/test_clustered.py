"""Small offline checks: python -m src.test_clustered."""
import tempfile
from pathlib import Path

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import ModernBertConfig, ModernBertForMaskedLM, PreTrainedTokenizerFast

from src.models.ClusteredTransformer import ClusteredTransformer, cluster_centroids, clustered_maxsim
from src.models.SparseEncoder import SparseEncoder
from src.loss.ClusteredLoss import ClusteredRankingLoss
from src.loss.SpladeLoss import SpladeMixedTopKLoss


def main():
    torch.manual_seed(0)
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp)
        tokenizer = Tokenizer(WordLevel(
            {"[UNK]": 0, "[PAD]": 1, "A": 2, "sample": 3, "longer": 4, "text": 5},
            unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]",
                                pad_token="[PAD]").save_pretrained(path)
        config = ModernBertConfig(
            vocab_size=6, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
            num_attention_heads=4, local_attention=16, pad_token_id=1,
            bos_token_id=0, eos_token_id=0, cls_token_id=0, sep_token_id=0,
            reference_compile=False,
        )
        ModernBertForMaskedLM(config).save_pretrained(path)
        torch.save(torch.arange(6) % 3, path / "cluster_ids.pt")
        for mode in ("splade", "colbert"):
            module = ClusteredTransformer(
                str(path), num_clusters=3, embedding_type=mode, max_seq_length=16,
                activation="log1p_relu", model_args={"attn_implementation": "sdpa"})
            model = SparseEncoder(modules=[module], device="cpu")
            features = [model.tokenize(["A sample", "A longer text"]),
                        model.tokenize(["sample text", "A text"])]
            similarity = {"similarity_fct": clustered_maxsim} if mode == "colbert" else {}
            loss = SpladeMixedTopKLoss(
                model=model, dense_loss=None, sparse_loss=ClusteredRankingLoss(model, **similarity),
                k=None, query_regularizer_weight=0.005, document_regularizer_weight=0.005)
            total = sum(loss(features).values())
            assert torch.isfinite(total)
            total.backward()
            assert module.w.grad is not None and torch.isfinite(module.w.grad).all()
            model.eval()
            with torch.no_grad():
                single = model(model.tokenize(["A sample"]))["sparse_embeddings"]
                padded = model(model.tokenize(["A sample", "A longer text"]))["sparse_embeddings"]
                torch.testing.assert_close(single[0], padded[0, :single.shape[1]] if mode == "colbert" else padded[0])
                if mode == "colbert":
                    expected = torch.einsum("ash,bth->abst", padded, padded).amax(-1).sum(-1)
                    expected /= (padded.abs().sum(-1) > 1e-6).sum(-1).clamp_min(1)[:, None]
                    torch.testing.assert_close(clustered_maxsim(padded, padded, 1), expected)
            saved = path / f"saved-{mode}"
            model.save_pretrained(str(saved), create_model_card=False)
            restored = SparseEncoder(str(saved), device="cpu")
            with torch.no_grad():
                actual = restored(restored.tokenize(["A sample"]))["sparse_embeddings"]
            torch.testing.assert_close(single, actual)
            centroids = cluster_centroids(module.w, module.cluster_ids, 3)
            expected_ids = (module.w @ centroids.T).argmax(-1)
            module.update_clusters(chunk_size=2)
            assert torch.equal(module.cluster_ids, expected_ids)
            print(f"{mode}: backward, padding, scoring, save/load, cluster updates passed")


if __name__ == "__main__":
    main()
