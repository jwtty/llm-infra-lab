"""GPU smoke test.

Runs a sustained matrix-multiplication loop on the GPU so that
``nvidia-smi`` reports non-trivial GPU-Util and memory usage.

Usage:
    python scripts/gpu_smoke_test.py [--size N] [--iters K] [--dtype fp16|fp32]

In a second terminal, run:
    watch -n 1 nvidia-smi
"""

from __future__ import annotations

import argparse
import time

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Simple GPU matmul smoke test")
    p.add_argument("--size", type=int, default=8192, help="square matrix side length")
    p.add_argument("--iters", type=int, default=200, help="number of matmul iterations")
    p.add_argument(
        "--dtype",
        choices=("fp16", "fp32"),
        default="fp16",
        help="tensor dtype; fp16 stresses tensor cores",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available. Check driver and PyTorch install.")

    device = torch.device("cuda:0")
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32

    print(f"Device : {torch.cuda.get_device_name(0)}")
    print(f"Dtype  : {dtype}")
    print(f"Size   : {args.size}x{args.size}")
    print(f"Iters  : {args.iters}")

    a = torch.randn(args.size, args.size, device=device, dtype=dtype)
    b = torch.randn(args.size, args.size, device=device, dtype=dtype)

    # Warm-up so the first timing isn't dominated by allocator / cuBLAS init.
    for _ in range(5):
        _ = a @ b
    torch.cuda.synchronize()

    start = time.perf_counter()
    c = a
    for i in range(args.iters):
        c = (a @ b) + 0.0001 * c  # keep result alive so it isn't optimized away
        if (i + 1) % 25 == 0:
            torch.cuda.synchronize()
            mem_mb = torch.cuda.memory_allocated() / (1024 ** 2)
            elapsed = time.perf_counter() - start
            print(f"  iter {i + 1:>4}/{args.iters}  "
                  f"elapsed={elapsed:6.2f}s  alloc={mem_mb:7.1f} MiB")

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    # Approximate TFLOPs: each matmul is 2 * N^3 FLOPs.
    flops = 2.0 * (args.size ** 3) * args.iters
    tflops = flops / elapsed / 1e12

    print(f"\nDone in {elapsed:.2f}s — ~{tflops:.2f} TFLOP/s sustained")
    print(f"Final result checksum: {c.float().sum().item():.4f}")


if __name__ == "__main__":
    main()
