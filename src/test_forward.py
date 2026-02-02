import torch
import argparse
from src.models.MLMTransformer import MLMTransformer
from src.models.pooling import SpladePooling, SparseColbertPooling
from src.models.SparseEncoder import SparseEncoder
from src.models.SparseColbertEncoder import SparseColbertEncoder
from src.models.pooling import LightSpladePooling
from src.kernels.fused_maxsim import sparse_maxsim, sparse_maxsim_pairwise

def test_sparse_encoder(model_name, device="cuda", backend="modernbert_fused_mean", act="log1p_relu"):
    print("\n--- Testing SparseEncoder (SPLADE) ---")
    
    # 1. Initialize modules
    mlm_transformer = MLMTransformer(
        model_name,
        max_seq_length=512,
        model_args={"attn_implementation": "flash_attention_2"},
        backend=backend
    )
    splade_pooling = LightSpladePooling(pooling_strategy="mean", activation_function=act)
    
    # 2. Initialize SparseEncoder
    model = SparseEncoder(modules=[mlm_transformer, splade_pooling])
    model.to(device)
    model.eval()
    if "bert" not in backend:
        model.bfloat16()
    
    # 3. Dummy Forward Pass
    sentences = ["This is a test sentence for SPLADE encoding.", "Another example."]
    
    print(f"Encoding {len(sentences)} sentences...")
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            # encode handles tokenization and forward pass
            embeddings = model.encode(sentences, convert_to_tensor=True)
    
    print(f"Output type: {type(embeddings)}")
    print(f"Output shape/size: {embeddings.shape if hasattr(embeddings, 'shape') else len(embeddings)}")
    
    # Show non-zero dimensions for the first sentence
    if hasattr(embeddings, "to_sparse"):
        sparse_emb = embeddings[0].to_sparse().coalesce()
        print(f"First sentence non-zero dimensions: {sparse_emb.indices().shape[1]}")
    
    return embeddings

from sentence_transformers.util import batch_to_device

def test_sparse_colbert_encoder(model_name, device="cuda", k=512, backend="modernbert_sparse_colbert", act="log1p_relu"):
    print("\n--- Testing SparseColbertEncoder (ColBERT) ---")
    
    # 1. Initialize modules
    mlm_transformer = MLMTransformer(
        model_name,
        max_seq_length=512,
        model_args={"attn_implementation": "flash_attention_2"},
        backend=backend
    )
    mlm_transformer.auto_model.k = k
    
    colbert_pooling = SparseColbertPooling(
        pooling_strategy="mean",
        activation_function=act
    )
    
    # 2. Initialize SparseColbertEncoder
    model = SparseColbertEncoder(
        modules=[mlm_transformer, colbert_pooling],
        embedding_type="colbert"
    )
    model.to(device)
    model.eval()
    if "bert" not in backend:
        model.bfloat16()
    
    # 3. Forward Pass (using logic from rerank_sparse.py)
    sentences = ["This is a test sentence for ColBERT encoding.", "Check sparse maxsim."]
    
    print(f"Encoding {len(sentences)} sentences...")
    with torch.no_grad():
        # Match rerank_sparse.py logic: tokenize then call model directly
        features = model.tokenize(sentences)
        features = batch_to_device(features, device)
        
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(features)
            vals, inds = output["sparse_embeddings"]
    
    print(f"Output Vals shape: {vals.shape}") # (Batch, Seq_Len, K)
    print(f"Output Inds shape: {inds.shape}") # (Batch, Seq_Len, K)
    
    # --- Testing Kernels ---
    print("\n--- Testing Similarity Kernels ---")
    
    # 1. Cross-similarity (sparse_maxsim)
    # Typically used for Q x D matching. 
    # Let's treat the first sentence as a query and both as a document batch.
    q = (vals[0:1], inds[0:1]) # First sentence as query (1, Tq, K)
    d = (vals, inds)           # Both as docs (B, Td, K)
    
    print(f"Testing sparse_maxsim (Query size {q[0].shape}, Doc batch {d[0].shape})...")
    scores = sparse_maxsim(q, d) # Returns (B1, B2) -> (1, 2)
    print(f"Scores (Q1 vs D1, D2): {scores}")

    # 2. Pairwise similarity (sparse_maxsim_pairwise)
    # Typically used for 1-to-1 matching (e.g. contrastive loss).
    print("\nTesting sparse_maxsim_pairwise (1-to-1 matching)...")
    q_pair = (vals, inds)
    d_pair = (vals, inds)
    pair_scores = sparse_maxsim_pairwise(q_pair, d_pair) # Returns (B,)
    print(f"Pairwise scores (diagonal): {pair_scores}")

    return vals, inds

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test forward pass for Sparse Encoders")
    parser.add_argument("--model_name", type=str, required=True, help="Path to the model checkpoint or HF model name")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run on")
    parser.add_argument("--type", type=str, choices=["splade", "colbert", "both"], default="both", help="Type of model to test")
    parser.add_argument("--k", type=int, default=512, help="K value for ColBERT")
    parser.add_argument("--backend", type=str, default="modernbert_sparse_colbert", help="Backend for ColBERT")
    parser.add_argument("--act", type=str, default="log1p_relu", help="Activation for ColBERT")
    
    args = parser.parse_args()
    
    if args.type in ["splade", "both"]:
        try:
            test_sparse_encoder(args.model_name, args.device, backend=args.backend, act=args.act)
        except Exception as e:
            print(f"SparseEncoder test failed: {e}")
            import traceback
            traceback.print_exc()
        
    if args.type in ["colbert", "both"]:
        try:
            test_sparse_colbert_encoder(args.model_name, args.device, k=args.k, backend=args.backend, act=args.act)
        except Exception as e:
            print(f"SparseColbertEncoder test failed: {e}")
            import traceback
            traceback.print_exc()
