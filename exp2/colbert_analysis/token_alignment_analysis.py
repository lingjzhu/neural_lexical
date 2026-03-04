import sys
import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer
import json
from tabulate import tabulate

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
    
    mapping_path = os.path.join(CURRENT_DIR, "results", "cluster_mapping.json")
    if not os.path.exists(mapping_path):
        print(f"❌ Error: cluster_mapping.json not found. Run cluster_analysis.py first.")
        return
        
    with open(mapping_path, "r") as f:
        cluster_mapping = json.load(f)

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
    
    samples = [
        "The quick brown fox jumps over the lazy dog.",
        "Clustered ColBERT uses sparse projections for efficient retrieval.",
        "A neural lexical search engine combines dense and sparse representations."
    ]
    
    results = []
    
    for text in samples:
        print(f"\n--- Analyzing: '{text}' ---")
        features = model.tokenize([text])
        features = {k: v.to("cuda") for k, v in features.items()}
        
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(features)
                # [1, SeqLen, Clusters]
                embeddings = outputs["sparse_embeddings"][0] 
        
        tokens = model.tokenizer.convert_ids_to_tokens(features["input_ids"][0])
        
        table_data = []
        for i, token in enumerate(tokens):
            if token == "[PAD]": continue
            
            vec = embeddings[i]
            top_vals, top_inds = torch.topk(vec, k=3)
            
            top_clusters_info = []
            for val, ind in zip(top_vals, top_inds):
                if val > 1e-6:
                    cid = str(int(ind))
                    mapped_tokens = cluster_mapping.get(cid, [])[:5]
                    top_clusters_info.append(f"C{cid}({val:.2f}): {','.join(mapped_tokens)}")
            
            table_data.append([token, "\n".join(top_clusters_info)])
            
        print(tabulate(table_data, headers=["Token", "Top Clusters (Score): Represented Tokens"], tablefmt="grid"))
        results.append({
            "text": text,
            "alignment": table_data
        })

    output_path = os.path.join(CURRENT_DIR, "results", "token_alignment.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Token alignment analysis saved to {output_path}")

if __name__ == "__main__":
    main()
