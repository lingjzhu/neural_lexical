# Neural Lexical Style Embeddings

This repository contains implementation for sparse models for authorship retrieval, including SPLADE and Sparse ColBERT, built on ModernBERT and Qwen. It also includes the lexicon-based SPLADE and SPARCOL models from [the paper](https://aclanthology.org/2026.findings-emnlp.400/).

## Installation

Ensure you have the following dependencies installed:

```bash
pip install torch "transformers>=4.57,<5" "sentence-transformers>=5.2,<6" flash-attn
# For initializing clusters in new lexicon training runs:
pip install datasets scikit-learn
```

## Checkpoints

We have released the following sparse retrieval models on the Hugging Face Hub:

| Model ID | Description |
| :--- | :--- |
| [UBC-SLIME/sparcol-large-k2048](https://huggingface.co/UBC-SLIME/sparcol-large-k2048) | Sparse ColBERT (ModernBERT-large) with K=2048 |
| [UBC-SLIME/sparcol-large-k1024](https://huggingface.co/UBC-SLIME/sparcol-large-k1024) | Sparse ColBERT (ModernBERT-large) with K=1024 |
| [UBC-SLIME/sparcol-large-k512-no-cls](https://huggingface.co/UBC-SLIME/sparcol-large-k512-no-cls) | Sparse ColBERT (ModernBERT-large) with K=512, no CLS token |
| [UBC-SLIME/splade-qwen3-0.6B-mean](https://huggingface.co/UBC-SLIME/splade-qwen3-0.6B-mean) | SPLADE (Qwen3-0.6B) with Mean Pooling |
| [UBC-SLIME/splade-base-mean](https://huggingface.co/UBC-SLIME/splade-base-mean) | SPLADE (ModernBERT-base) with Mean Pooling |
| [UBC-SLIME/splade-large-mean](https://huggingface.co/UBC-SLIME/splade-large-mean) | SPLADE (ModernBERT-large) with Mean Pooling |

The final lexicon checkpoints use ModernBERT-large. Cluster assignments are included
with each complete checkpoint and must be loaded alongside its weights.

| Model ID | Description | Final step |
| :--- | :--- | :--- |
| [UBC-SLIME/splade-lexicon-large-c4000](https://huggingface.co/UBC-SLIME/splade-lexicon-large-c4000) | SPLADE-Lexicon, 4,000 clusters, fused mean pooling | 1582 |
| [UBC-SLIME/sparcol-lexicon-large-c500](https://huggingface.co/UBC-SLIME/sparcol-lexicon-large-c500) | SPARCOL-Lexicon, 500 clusters, no CLS/SEP | 1580 |
| [UBC-SLIME/sparcol-lexicon-large-c1000](https://huggingface.co/UBC-SLIME/sparcol-lexicon-large-c1000) | SPARCOL-Lexicon, 1,000 clusters, no CLS/SEP | 1580 |
| [UBC-SLIME/sparcol-lexicon-large-c2000](https://huggingface.co/UBC-SLIME/sparcol-lexicon-large-c2000) | SPARCOL-Lexicon, 2,000 clusters, no CLS/SEP | 1580 |

The local 8,000-cluster SPLADE checkpoint is missing its learned `cluster_ids.pt`.
Its original clustering cannot be recovered exactly from the weights or initial k-means
assignments, so a complete inference checkpoint is not released for that variant.
Each published checkpoint has a `checkpoint_info.json` recording its source run and step.

## Usage

The following examples demonstrate how to inference with the sparse encoders.

### 1. SPLADE (SparseEncoder)

The `SparseEncoder` handles tokenization and SPLADE encoding.

```python
import torch
from src.models.MLMTransformer import MLMTransformer
from src.models.pooling import LightSpladePooling
from src.models.SparseEncoder import SparseEncoder

# 1. Initialize modules
mlm_transformer = MLMTransformer(
    "UBC-SLIME/splade-large-mean",
    max_seq_length=512,
    model_args={"attn_implementation": "flash_attention_2"},
    backend="modernbert_fused_mean"
)
splade_pooling = LightSpladePooling(pooling_strategy="mean", activation_function="log1p_relu")

# 2. Initialize SparseEncoder
model = SparseEncoder(modules=[mlm_transformer, splade_pooling])
model.to("cuda")
model.eval()

# 3. Encoding
sentences = ["This is a test sentence for SPLADE encoding.", "Another example."]
with torch.no_grad():
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        embeddings = model.encode(sentences, convert_to_tensor=True)

print(f"Output shape: {embeddings.shape}")
# you can use dot product to compute similarity
```

### 2. Sparse ColBERT (SparseColbertEncoder)

The `SparseColbertEncoder` represents sentences as a sequence of sparse vectors.

```python
import torch
from sentence_transformers.util import batch_to_device
from src.models.MLMTransformer import MLMTransformer
from src.models.pooling import SparseColbertPooling
from src.models.SparseColbertEncoder import SparseColbertEncoder

# 1. Initialize modules
mlm_transformer = MLMTransformer(
    "UBC-SLIME/sparcol-large-k512-no-cls",
    max_seq_length=512,
    model_args={"attn_implementation": "flash_attention_2"},
    backend="modernbert_sparse_colbert",
    add_special_token=False # if you don't need the CLS token, you can set this to False and use the checkpoint without the CLS token
)
mlm_transformer.auto_model.k = 512 # Set K sparsity

colbert_pooling = SparseColbertPooling(
    pooling_strategy="mean",
    activation_function="log1p_relu"
)

# 2. Initialize SparseColbertEncoder
model = SparseColbertEncoder(
    modules=[mlm_transformer, colbert_pooling],
    embedding_type="colbert"
)
model.to("cuda")
model.eval()

# 3. Forward Pass
sentences = ["This is a test sentence for ColBERT encoding.", "Check sparse maxsim."]
features = model.tokenize(sentences)
features = batch_to_device(features, "cuda")

with torch.no_grad():
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(features)
        vals, inds = output["sparse_embeddings"]

print(f"Output Vals shape: {vals.shape}") # (Batch, Seq_Len, K)
```

The `src.kernels.fused_maxsim` provides optimized kernels for computing MaxSim scores between sparse embeddings from Sparse ColBERT.

```python
from src.kernels.fused_maxsim import sparse_maxsim, sparse_maxsim_pairwise

# q and d are tuples of (vals, inds)
q = (q_vals, q_inds)
d = (d_vals, d_inds)


# computing the similarity is really slow, so a triton kernel is used. it is not as fast as it should be, but it is fast enough for now. if you need scores for specific tokens, you can subset the sequence before calling the kernel.

# 1. Cross-similarity (B1 x B2)
# Computes similarity between all pairs of queries and documents
cross_scores = sparse_maxsim(q, d) 

# 2. Pairwise-similarity (B)
# Computes similarity between corresponding pairs (q[i] vs d[i])
# scores are averaged across the sequence. if you want just the sum, you can set normalize=False. Note that disabling normalization will make the score uncomparable across queries of different length. if you only have one query, you can set normalize=False.
pair_scores = sparse_maxsim_pairwise(q, d)
```

### 3. Lexicon models (ClusteredTransformer)

The shared module loads learned clusters and the pooling/tokenization settings saved
with the checkpoint. Run these examples from the repository root.

```python
import torch
from src.models.ClusteredTransformer import ClusteredTransformer, clustered_maxsim
from src.models.SparseEncoder import SparseEncoder

module = ClusteredTransformer.load("UBC-SLIME/splade-lexicon-large-c4000")
model = SparseEncoder(modules=[module]).to("cuda").eval()
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    embeddings = model.encode(["A sample document.", "Another document."],
                              convert_to_tensor=True, convert_to_sparse_tensor=False)
    scores = embeddings @ embeddings.T
```

For SPARCOL-Lexicon, keep the token dimension and score with mean MaxSim:

```python
module = ClusteredTransformer.load("UBC-SLIME/sparcol-lexicon-large-c2000")
model = SparseEncoder(modules=[module]).to("cuda").eval()
features = model.tokenize(["A sample document.", "Another document."])
features = {k: v.to("cuda") for k, v in features.items()}
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    embeddings = model(features)["sparse_embeddings"]  # (batch, tokens, clusters)
    scores = clustered_maxsim(embeddings, embeddings, chunk_size=8)
```

These cluster vectors use dense storage and remain unnormalized. Zero vectors are
excluded from the query-length denominator in MaxSim. The SPLADE checkpoint uses
ReLU, mean pooling, then two `log1p` operations, matching its fused training path.
Both the fused and PyTorch paths preserve that ordering. For CPU inference, set
`module.use_triton = False` and move the model and inputs to CPU.

### Training lexicon models

Training data is JSONL with `anchor` (or `query`) and `positive` text fields.
Initialize token clusters with k-means, then greedily update assignments every ten
optimizer steps. The module saves learned assignments with each checkpoint.

```bash
python -m src.train_clustered \
    --train_data data/training_data_v1_final_dedup.jsonl \
    --output_dir outputs/splade-lexicon-c4000 \
    --embedding_type splade --num_clusters 4000 --activation log1p_relu --use_triton

python -m src.train_clustered \
    --train_data data/training_data_v1_final_dedup.jsonl \
    --output_dir outputs/sparcol-lexicon-c2000 \
    --embedding_type colbert --num_clusters 2000 --activation relu
```

Use `torchrun --nproc_per_node=4 -m src.train_clustered ...` for distributed training;
`--batch_size` is per device. This compact entry point uses full minibatches; lower
`--batch_size` to fit available GPU memory. It does not include the experimental
training variants or gradient caching from the development branch.

## Running Tests

You can run the forward pass test script directly:

```bash
python src/test_forward.py --model_name path/to/checkpoint --type both
```


Run the small offline lexicon checks (gradients, padding, MaxSim, cluster updates,
and checkpoint save/load) with:

```bash
python -m src.test_clustered
```
