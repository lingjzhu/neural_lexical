import torch
import torch.nn as nn
from sentence_transformers import util
from typing import Iterable, dict
import sys
import os

# Add parent dir to path to import src
sys.path.append(os.path.abspath(".."))

from src.models.SparseEncoder import SparseEncoder
from src.models.MLMTransformer import MLMTransformer
from src.loss.SpladeLoss import CachedSpladeMixedTopKLoss, SparseSelfMultipleNegativesRankingLoss, FlopsLoss

def test_cache_loss():
    print("Initializing model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    transformer = MLMTransformer("answerdotai/ModernBERT-base", model_type="modernbert-base")
    model = SparseEncoder(transformer).to(device)
    
    print("Setting up loss...")
    sparse_loss = SparseSelfMultipleNegativesRankingLoss(model=model, scale=1.0)
    loss_fn = CachedSpladeMixedTopKLoss(
        model=model,
        dense_loss=None,
        sparse_loss=sparse_loss,
        mini_batch_size=2,
        k=[100],
        document_regularizer_weight=0.01,
        query_regularizer_weight=0.01,
    ).to(device)

    # Fake data: 4 items (mini_batch_size=2 means 2 minibatches)
    # 3 sentences (query, pos, neg)
    batch_size = 4
    sentence_features = []
    for i in range(3):
        feat = {
            "input_ids": torch.randint(0, 1000, (batch_size, 10), device=device),
            "attention_mask": torch.ones((batch_size, 10), device=device)
        }
        sentence_features.append(feat)

    print("Running forward pass...")
    losses = loss_fn(sentence_features)
    print(f"Losses: {[(k, v.item()) for k, v in losses.items()]}")

    total_loss = losses["sparse_loss"]
    print("Running backward pass...")
    total_loss.backward()
    print("Backward pass complete!")

if __name__ == "__main__":
    test_cache_loss()
