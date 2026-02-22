import torch
import time
import argparse
from clustered_splade import UnembeddingCompressSparseFunction
from triton_kernels import fused_sim_argmax


def benchmark_memory_and_time(func, *args, num_iters=10, num_warmup=1):
    # Warmup
    for _ in range(num_warmup):
        res = func(*args)
        if isinstance(res, tuple):
            out = res[0]
        else:
            out = res
        if out.requires_grad:
            out.sum().backward()
            
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(num_iters):
        res = func(*args)
        if isinstance(res, tuple):
            out = res[0]
        else:
            out = res
            
        if out.requires_grad:
            out.sum().backward()
    end_event.record()
    
    torch.cuda.synchronize()
    
    elapsed_time_ms = start_event.elapsed_time(end_event) / num_iters
    peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    
    return elapsed_time_ms, peak_memory_mb, res

def check_precision(out_pt, out_tr, name, atol=1e-3, rtol=1e-3):
    diff = torch.abs(out_pt - out_tr)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    is_close = torch.allclose(out_pt, out_tr, atol=atol, rtol=rtol)
    print(f"[{name}] Max diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f}, All close: {is_close}")
    return is_close

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--B", type=int, default=32)
    parser.add_argument("--S", type=int, default=512)
    parser.add_argument("--H", type=int, default=768)
    parser.add_argument("--V", type=int, default=150000)
    parser.add_argument("--C", type=int, default=8000)
    args = parser.parse_args()

    for dtype in [torch.float32, torch.bfloat16]:
        print(f"\n=======================")
        print(f"Benchmarking with {dtype}")
        print(f"Shapes: H=[{args.B}, {args.S}, {args.H}], W=[{args.V}, {args.H}], Clusters={args.C}")
        print(f"=======================")
        
        device = 'cuda'
        h = torch.randn(args.B, args.S, args.H, device=device, dtype=dtype, requires_grad=True)
        w = torch.randn(args.V, args.H, device=device, dtype=dtype, requires_grad=True)
        cluster_ids = torch.randint(0, args.C, (args.V,), device=device, dtype=torch.long)
        
        # 1. Forward/Backward
        def run_pytorch():
            if h.grad is not None: h.grad.zero_()
            if w.grad is not None: w.grad.zero_()
            return UnembeddingCompressSparseFunction.apply(h, w, cluster_ids, args.C, False)
            
        def run_triton():
            if h.grad is not None: h.grad.zero_()
            if w.grad is not None: w.grad.zero_()
            return UnembeddingCompressSparseFunction.apply(h, w, cluster_ids, args.C, True)

        # Precision Check
        out_pt = run_pytorch()
        out_pt.sum().backward()
        h_grad_pt = h.grad.clone()
        w_grad_pt = w.grad.clone()
        
        out_tr = run_triton()
        out_tr.sum().backward()
        h_grad_tr = h.grad.clone()
        w_grad_tr = w.grad.clone()
        
        check_precision(out_pt, out_tr, "Forward")
        check_precision(h_grad_pt, h_grad_tr, "Backward H")
        check_precision(w_grad_pt, w_grad_tr, "Backward W")

        # Benchmarking
        print("Running PyTorch benchmark...")
        pt_time, pt_mem, _ = benchmark_memory_and_time(run_pytorch)
        print("Running Triton benchmark...")
        tr_time, tr_mem, _ = benchmark_memory_and_time(run_triton)
        
        print(f"\nForward + Backward Performance:")
        print(f"PyTorch: {pt_time:.2f} ms | {pt_mem:.2f} MB")
        print(f"Triton : {tr_time:.2f} ms | {tr_mem:.2f} MB")
        print(f"Speedup: {pt_time/tr_time:.2f}x")
        
        # 2. Update Mask (Sim Argmax)
        w_reduced = torch.randn(args.C, args.H, device=device, dtype=dtype)
        
        def run_pytorch_argmax():
            sims = torch.matmul(w, w_reduced.t())
            return sims.argmax(dim=-1)
            
        def run_triton_argmax():
            return fused_sim_argmax(w, w_reduced)

        out_argmax_pt = run_pytorch_argmax()
        out_argmax_tr = run_triton_argmax()
        
        matches = (out_argmax_pt == out_argmax_tr).float().mean().item()
        print(f"\n[Update Mask] Precision matches: {matches*100:.2f}%")
        
        pt_time_am, pt_mem_am, _ = benchmark_memory_and_time(run_pytorch_argmax)
        tr_time_am, tr_mem_am, _ = benchmark_memory_and_time(run_triton_argmax)
        
        print(f"Update Mask Performance:")
        print(f"PyTorch: {pt_time_am:.2f} ms | {pt_mem_am:.2f} MB")
        print(f"Triton : {tr_time_am:.2f} ms | {tr_mem_am:.2f} MB")
        print(f"Speedup: {pt_time_am/tr_time_am:.2f}x\n")

        # 3. Fused Pooling Performance
        print(f"=======================")
        print(f"Fused Pooling Benchmarking")
        print(f"=======================")
        
        import sys
        import os
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
        from kernels.fused_sumpool import FeatureCountFunction
        
        def run_fused_pytorch():
            if h.grad is not None: h.grad.zero_()
            if w.grad is not None: w.grad.zero_()
            # Full sequence literally mapped in PyTorch: index_add -> matmul -> relu -> sum
            wr = torch.zeros(args.C, args.H, device=device, dtype=dtype)
            wr.index_add_(0, cluster_ids, w)
            logits = torch.matmul(h, wr.t()) # Materializes [B, S, C] tensor
            out = torch.sum(torch.relu(logits), dim=1)
            return out
            
        def run_fused_triton_reused():
            if h.grad is not None: h.grad.zero_()
            if w.grad is not None: w.grad.zero_()
            wr = torch.zeros(args.C, args.H, device=device, dtype=dtype)
            wr.index_add_(0, cluster_ids, w)
            return FeatureCountFunction.apply(h, wr) # Computes ReLU/Sum without intermediate [B, S, C]

        # Precision Check
        out_f_pt = run_fused_pytorch()
        out_f_pt.sum().backward()
        h_grad_f_pt = h.grad.clone()
        w_grad_f_pt = w.grad.clone()
        
        out_f_tr = run_fused_triton_reused()
        out_f_tr.sum().backward()
        h_grad_f_tr = h.grad.clone()
        w_grad_f_tr = w.grad.clone()
        
        check_precision(out_f_pt, out_f_tr, "Fused Forward (Reused Triton)")
        check_precision(h_grad_f_pt, h_grad_f_tr, "Fused Backward H (Reused Triton)")
        check_precision(w_grad_f_pt, w_grad_f_tr, "Fused Backward W (Reused Triton)")

        print("Running Fused PyTorch (Materializes Intermediate)...")
        f_pt_time, f_pt_mem, _ = benchmark_memory_and_time(run_fused_pytorch, num_iters=5)
        print("Running Fused Triton (FeatureCountFunction NO Intermediate)...")
        f_tr_time, f_tr_mem, _ = benchmark_memory_and_time(run_fused_triton_reused, num_iters=5)
        
        print(f"\nFused Forward + Backward Performance:")
        print(f"PyTorch         : {f_pt_time:.2f} ms | {f_pt_mem:.2f} MB")
        print(f"Triton (Reused) : {f_tr_time:.2f} ms | {f_tr_mem:.2f} MB")
        print(f"Speedup vs PT   : {f_pt_time/f_tr_time:.2f}x\n")

if __name__ == "__main__":
    main()
