import torch
import argparse
from src.models.MLMTransformer import MLMTransformer
from src.models.pooling import LightSpladePooling
from src.models.SparseEncoder import SparseEncoder
from sentence_transformers.util import batch_to_device

def verify_splade_encode(model_name, device="cuda", backend="modernbert_fused_mean", act="log1p_relu"):
    print(f"--- Verifying SparseEncoder.encode vs Manual Forward Pass (SPLADE) ---")
    
    # 1. Initialize model
    mlm_transformer = MLMTransformer(
        model_name,
        max_seq_length=512,
        model_args={"attn_implementation": "flash_attention_2"},
        backend=backend
    )
    splade_pooling = LightSpladePooling(pooling_strategy="mean", activation_function=act)
    model = SparseEncoder(modules=[mlm_transformer, splade_pooling])
    model.to(device)
    model.eval()
    if "bert" not in backend:
        model.bfloat16()
    
    sentences = ["This is a test sentence.", "Verification of SPLADE encode method."]
    
    # 2. Manual Forward Pass
    print("Performing manual forward pass...")
    with torch.no_grad():
        features = model.tokenize(sentences)
        features = batch_to_device(features, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(features)
            manual_embeddings = output["sparse_embeddings"]
    
    # 3. model.encode()
    print("Performing model.encode()...")
    with torch.no_grad():
        # current encode doesn't have autocast
        encode_embeddings = model.encode(sentences, convert_to_tensor=True, device=device)
    
    # 4. Compare
    print(f"Manual shape: {manual_embeddings.shape}, Encode shape: {encode_embeddings.shape}")
    
    # convert to dense if sparse
    if manual_embeddings.is_sparse:
        manual_embeddings = manual_embeddings.to_dense()
    if encode_embeddings.is_sparse:
        encode_embeddings = encode_embeddings.to_dense()
        
    vals_match = torch.allclose(manual_embeddings, encode_embeddings.to(manual_embeddings.dtype), atol=1e-5)
    print(f"Values match: {vals_match}")
    
    if not vals_match:
        diff = (manual_embeddings - encode_embeddings.to(manual_embeddings.dtype)).abs().max()
        print(f"Max value difference: {diff.item()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--backend", type=str, default="modernbert_fused_mean")
    args = parser.parse_args()
    
    verify_splade_encode(args.model_name, args.device, backend=args.backend)
