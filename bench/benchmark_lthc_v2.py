from __future__ import annotations

import argparse
import time

import torch

from flowmatching_lthc.models import build_model


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def benchmark_model(
    name: str,
    batch_size: int,
    warmup: int,
    iters: int,
    compile_mode: str | None,
    attn_backend: str,
) -> dict[str, float | str]:
    torch.manual_seed(123)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")
    model = build_model(name, attn_backend=attn_backend).to(device=device, dtype=torch.bfloat16).train()
    if compile_mode:
        model = torch.compile(model, mode=compile_mode)
    opt = torch.optim.SGD(model.parameters(), lr=0.0)
    x = torch.randn(batch_size, 3, 256, 256, device=device, dtype=torch.bfloat16)
    target = torch.randn_like(x)
    t = torch.rand(batch_size, device=device)
    y = torch.randint(0, 1000, (batch_size,), device=device)

    def step() -> torch.Tensor:
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = model(x, t, y)
            loss = (pred.float() - target.float()).square().mean()
        loss.backward()
        opt.step()
        return loss

    torch.cuda.reset_peak_memory_stats()
    for _ in range(warmup):
        step()
    synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        loss = step()
    synchronize()
    elapsed = time.perf_counter() - start
    step_s = elapsed / iters
    return {
        "model": name,
        "batch_size": batch_size,
        "compile": compile_mode or "none",
        "step_ms": step_s * 1000.0,
        "samples_per_s": batch_size / step_s,
        "peak_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "loss": float(loss.detach()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--compile_mode", default="none", choices=["none", "default", "reduce-overhead", "max-autotune"])
    parser.add_argument("--attn_backend", default="flash")
    args = parser.parse_args()

    compile_mode = None if args.compile_mode == "none" else args.compile_mode
    for name in args.models:
        torch.cuda.empty_cache()
        result = benchmark_model(name, args.batch_size, args.warmup, args.iters, compile_mode, args.attn_backend)
        print(
            f"{result['model']} bs={result['batch_size']} compile={result['compile']} "
            f"step_ms={result['step_ms']:.2f} img_s={result['samples_per_s']:.1f} "
            f"peak_gib={result['peak_gib']:.2f} loss={result['loss']:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
