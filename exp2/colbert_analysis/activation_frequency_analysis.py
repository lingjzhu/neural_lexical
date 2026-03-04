import sys
import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer
import json
from tqdm import tqdm
from collections import Counter

# Environment setup same as training/reranking
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
os.environ["HF_TRUST_REMOTE_CODE"] = "1"
os.environ["TRUST_REMOTE_CODE"] = "True"

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

# Add project root to sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '../..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.models.ClusteredColbertEncoder import ClusteredColbertEncoder
from exp2.train_clustered_colbert import ClusteredColbertTransformer

def main():
    checkpoint_path = "/home/slimelab/Projects/neural_lexical/exp2/outputs_clustered_colbert_modernbert_large_2000_relu/checkpoint-1580"
    num_clusters = 2000
    model_type = "modern-bert-large"
    activation = "relu"
    data_path = "/home/slimelab/Projects/neural_lexical/data/amazon_triplets.jsonl"
    num_samples = 1000
    batch_size = 32

    print(f"🔹 Loading model from {checkpoint_path}...")
    mlm_transformer = ClusteredColbertTransformer(
        checkpoint_path,
        max_seq_length=512,
        model_args={"attn_implementation": "sdpa"},
        model_type=model_type,
        num_clusters=num_clusters,
        activation=activation
    )
    
    model = ClusteredColbertEncoder(
        modules=[mlm_transformer],
        embedding_type="colbert"
    )
    model.to("cuda")
    model.eval()
    if "modernbert" in checkpoint_path.lower():
        model.to(torch.bfloat16)

    print(f"🔹 Loading {num_samples} articles from {data_path}...")
    articles = []
    if os.path.exists(data_path):
        with open(data_path, "r") as f:
            for i, line in enumerate(f):
                if i >= num_samples:
                    break
                obj = json.loads(line)
                articles.append(obj.get("positive", ""))
    else:
        print(f"⚠️ data_path {data_path} not found. Using dummy data for testing.")
        articles = ["This is a test article."] * 10
        num_samples = 10

    cluster_counts = Counter()
    token_cluster_counts = Counter()

    print(f"🔹 Running inference on {len(articles)} articles...")
    for i in tqdm(range(0, len(articles), batch_size)):
        batch_texts = articles[i:i+batch_size]
        features = model.tokenize(batch_texts)
        features = {k: v.to("cuda") for k, v in features.items()}
        
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(features)
                # [Batch, SeqLen, Clusters]
                embeddings = outputs["sparse_embeddings"] 
        
        # 1. Document-level frequency: Is cluster active ANYWHERE in the document?
        # Max over sequence dimension: [Batch, Clusters]
        doc_max = embeddings.max(dim=1).values
        doc_active = doc_max > 1e-6
        
        # 2. Token-level frequency: How many token positions activate this cluster?
        token_active = embeddings > 1e-6
        
        # Update counters
        active_indices = doc_active.nonzero(as_tuple=False)
        cluster_counts.update(active_indices[:, 1].tolist())
        
        token_active_indices = token_active.nonzero(as_tuple=False)
        token_cluster_counts.update(token_active_indices[:, 2].tolist())

    # Save results
    output_dir = os.path.join(CURRENT_DIR, "results")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save overall document-level frequencies
    with open(os.path.join(output_dir, "frequencies_doc.json"), "w") as f:
        json.dump({str(k): v for k, v in cluster_counts.items()}, f, indent=2)
        
    # Save overall token-level frequencies
    with open(os.path.join(output_dir, "frequencies_token.json"), "w") as f:
        json.dump({str(k): v for k, v in token_cluster_counts.items()}, f, indent=2)

    print(f"✅ Frequency analysis complete. Results saved in {output_dir}")

if __name__ == "__main__":
    main()
