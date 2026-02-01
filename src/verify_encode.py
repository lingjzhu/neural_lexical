import torch
import argparse
from src.models.MLMTransformer import MLMTransformer
from src.models.pooling import SparseColbertPooling
from src.models.SparseColbertEncoder import SparseColbertEncoder
from sentence_transformers.util import batch_to_device

def verify_encode(model_name, device="cuda", k=512, backend="modernbert_sparse_colbert", act="log1p_relu"):
    print(f"--- Verifying SparseColbertEncoder.encode vs Manual Forward Pass ---")
    
    # 1. Initialize model
    mlm_transformer = MLMTransformer(
        model_name,
        max_seq_length=512,
        model_args={"attn_implementation": "flash_attention_2"},
        backend=backend
    )
    mlm_transformer.auto_model.k = k
    colbert_pooling = SparseColbertPooling(pooling_strategy="mean", activation_function=act)
    model = SparseColbertEncoder(modules=[mlm_transformer, colbert_pooling], embedding_type="colbert")
    model.to(device)
    model.eval()
    if "bert" not in backend:
        model.bfloat16()
    
    sentences = ["This is a test sentence.", "Verification of encode method."]
    
    # 2. Manual Forward Pass (the "ground truth" from rerank_sparse.py)
    print("Performing manual forward pass...")
    with torch.no_grad():
        features = model.tokenize(sentences)
        features = batch_to_device(features, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(features)
            manual_vals, manual_inds = output["sparse_embeddings"]
    
    # 3. model.encode()
    print("Performing model.encode()...")
    with torch.no_grad():
        # We need to see if model.encode uses autocast internally (it doesn't yet)
        encode_vals, encode_inds = model.encode(sentences, return_tensors=True, device=device)
    
    # 4. Compare
    print(f"Manual Vals shape: {manual_vals.shape}, Encode Vals shape: {encode_vals.shape}")
    print(f"Manual Inds shape: {manual_inds.shape}, Encode Inds shape: {encode_inds.shape}")
    
    vals_match = torch.allclose(manual_vals, encode_vals.to(manual_vals.dtype), atol=1e-5)
    inds_match = torch.equal(manual_inds, encode_inds)
    
    print(f"Values match: {vals_match}")
    print(f"Indices match: {inds_match}")
    
    if not vals_match:
        diff = (manual_vals - encode_vals.to(manual_vals.dtype)).abs().max()
        print(f"Max value difference: {diff.item()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--k", type=int, default=512)
    parser.add_argument("--backend", type=str, default="modernbert_sparse_colbert")
    args = parser.parse_args()
    
    verify_encode(args.model_name, args.device, k=args.k, backend=args.backend)
