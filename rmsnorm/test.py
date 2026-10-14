import argparse
import time

import torch

try:
    from .custom import rms_norm as custom_rms_norm
    from .original import RMSNorm, rms_norm as tilelang_rms_norm
except ImportError:
    from custom import rms_norm as custom_rms_norm
    from original import RMSNorm, rms_norm as tilelang_rms_norm


def benchmark(fn, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1_000_000 / iterations


def main() -> None:
    parser = argparse.ArgumentParser(description="Test RMSNorm implementations.")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This test requires a CUDA-capable GPU.")

    torch.manual_seed(0)
    dtype = torch.bfloat16
    x = torch.randn(args.batch, args.seq_len, args.dim, device="cuda", dtype=dtype)
    weight = torch.randn(args.dim, device="cuda", dtype=dtype)
    torch_reference = RMSNorm(args.dim).to(device="cuda", dtype=dtype)
    torch_reference.weight.data.copy_(weight)

    # The original TileLang kernel normalizes 2D float32 inputs and has no weight.
    x_2d = x.reshape(-1, args.dim).float()
    tilelang_original = lambda: tilelang_rms_norm(x_2d, 1).reshape_as(x) * weight
    @torch.inference_mode()
    def torch_baseline():
        return torch_reference(x)
    tilelang_custom = lambda: custom_rms_norm(
        batch=args.batch,
        seq_len=args.seq_len,
        dim=args.dim,
        eps=torch_reference.eps,
        dtype=str(dtype).removeprefix("torch."),
    )(x, weight)

    reference = torch_baseline()
    original_output = tilelang_original()
    custom_output = tilelang_custom()
    torch.testing.assert_close(original_output, reference.float(), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(custom_output.float(), reference.float(), rtol=2e-2, atol=2e-2)

    measurements = {
        "torch": benchmark(torch_baseline, args.warmup, args.iterations),
        "original_tilelang": benchmark(tilelang_original, args.warmup, args.iterations),
        "custom_tilelang": benchmark(tilelang_custom, args.warmup, args.iterations),
    }
    for name, latency_us in measurements.items():
        print(f"{name}: {latency_us:.2f} us")


if __name__ == "__main__":
    main()
