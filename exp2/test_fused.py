import torch
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from kernels.fused_sumpool import FeatureCountFunction

def test():
    B, S, H, V, C = 32, 512, 768, 36000, 8000
    device = 'cuda'
    dtype = torch.bfloat16
    
    h = torch.randn(B, S, H, device=device, dtype=dtype, requires_grad=True)
    w = torch.randn(V, H, device=device, dtype=dtype, requires_grad=True)
    cluster_ids = torch.randint(0, C, (V,), device=device, dtype=torch.long)
    
    print("Running PyTorch Reference...")
    # Reference
    w_reduced = torch.zeros(C, H, device=device, dtype=dtype)
    w_reduced.index_add_(0, cluster_ids, w)
    logits = torch.matmul(h, w_reduced.t())
    out_pt = torch.sum(torch.relu(logits), dim=1)
    out_pt.sum().backward()
    h_grad_pt = h.grad.clone()
    
    print("Running Triton Fused...")
    h.grad.zero_()
    out_tr = FeatureCountFunction.apply(h, w_reduced)
    out_tr.sum().backward()
    h_grad_tr = h.grad.clone()
    
    diff_fwd = (out_pt - out_tr).abs().max().item()
    diff_bwd = (h_grad_pt - h_grad_tr).abs().max().item()
    
    print(f"Forward diff: {diff_fwd:.6f}")
    print(f"Backward H diff: {diff_bwd:.6f}")

if __name__ == "__main__":
    test()
