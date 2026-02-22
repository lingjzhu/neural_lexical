import sys
import os
print("Importing torch...")
import torch
sys.path.append(os.path.abspath(".."))
print("Importing SparseEncoder...")
from src.models.SparseEncoder import SparseEncoder
print("Importing MLMTransformer...")
from src.models.MLMTransformer import MLMTransformer
print("Importing SpladeLoss...")
from src.loss.SpladeLoss import CachedSpladeMixedTopKLoss
print("Import successful!")
print("Instantiating MLMTransformer...")
from src.models.MLMTransformer import MLMTransformer
print("Instantiating ClusteredMLMTransformer...")
from train_clustered_splade import ClusteredMLMTransformer
from src.loss.SpladeLoss import CachedSpladeMixedTopKLoss, SparseSelfMultipleNegativesRankingLoss
# Use small clusters for speed
model = ClusteredMLMTransformer("answerdotai/ModernBERT-base", num_clusters=128, model_type="modernbert-base")

print("Initializing SparseEncoder...")
sparse_encoder = SparseEncoder(modules=[model]).to("cuda")

# Fake data
print("Creating fake data...")
batch_size = 4
sentence_features = []
for i in range(3):
    feat = {
        "input_ids": torch.randint(0, 50000, (batch_size, 128), device="cuda"),
        "attention_mask": torch.ones((batch_size, 128), device="cuda", dtype=torch.bool)
    }
    sentence_features.append(feat)

print("Initializing CachedSpladeMixedTopKLoss...")
sparse_loss = SparseSelfMultipleNegativesRankingLoss(model=sparse_encoder, scale=1.0)
loss_fn = CachedSpladeMixedTopKLoss(
    model=sparse_encoder,
    dense_loss=None,
    sparse_loss=sparse_loss,
    mini_batch_size=2,
    k=[100],
    document_regularizer_weight=0.01,
    query_regularizer_weight=0.01,
).to("cuda")

print("Running forward pass...")
losses = loss_fn(sentence_features)
print(f"Losses: {[(k, v.item()) for k, v in losses.items()]}")

print("Running backward pass...")
losses["sparse_loss"].backward()
print("Backward successful!")
