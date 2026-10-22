"""Correctness and benchmarks for SANA LiteLA attention and end-to-end paths."""

import argparse
import sys
import time
from pathlib import Path

import torch

if __package__:
    from .custom import (
        CUDAGraphJointLiteLA,
        CustomOpJointLiteLA,
        JointLiteLA,
        optimized_joint_litela,
        qkv_attn_matmul,
    )
    from .original import LiteLA as OriginalLiteLA
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from SANA_LinearAttention.custom import (
        CUDAGraphJointLiteLA,
        CustomOpJointLiteLA,
        JointLiteLA,
        optimized_joint_litela,
        qkv_attn_matmul,
    )
    from SANA_LinearAttention.original import LiteLA as OriginalLiteLA


DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def benchmark(fn, warmup: int, iterations: int) -> float:
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iterations):
            fn()
        torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1_000_000 / iterations


def test_attn_matmul(args, seq_len: int, channels: int, dtype: torch.dtype) -> None:
    torch._dynamo.reset()
    heads = channels // args.head_dim
    qkv = torch.randn(args.batch, seq_len, 3, channels, device="cuda", dtype=dtype)
    q, k, v = qkv.unbind(2)
    q = q.transpose(-1, -2).reshape(args.batch, heads, args.head_dim, seq_len)
    k = k.transpose(-1, -2).reshape(args.batch, heads, args.head_dim, seq_len).transpose(-1, -2)
    v = v.transpose(-1, -2).reshape(args.batch, heads, args.head_dim, seq_len)

    original = OriginalLiteLA(
        channels, channels, heads=heads, dim=args.head_dim, qk_norm=False
    ).cuda().to(dtype).eval()
    eager_original = original.attn_matmul
    compiled_original = torch.compile(original.attn_matmul, mode="max-autotune")
    custom = qkv_attn_matmul(
        args.batch,
        heads,
        args.head_dim,
        seq_len,
        dtype="float" if dtype == torch.float32 else args.dtype,
        output_dtype="float" if dtype == torch.float32 else args.dtype,
    )

    with torch.inference_mode():
        eager_expected = eager_original(q, k, v).view(args.batch, channels, seq_len)
        eager_expected = eager_expected.permute(0, 2, 1).clone()
        compiled_expected = compiled_original(q, k, v).view(args.batch, channels, seq_len)
        compiled_expected = compiled_expected.permute(0, 2, 1).clone()
        actual = custom(qkv).clone()
    torch.testing.assert_close(
        compiled_expected.float(), eager_expected.float(), rtol=2e-2, atol=2e-2
    )
    torch.testing.assert_close(actual.float(), eager_expected.float(), rtol=2e-2, atol=2e-2)
    max_error = (actual.float() - eager_expected.float()).abs().max().item()
    eager_us = benchmark(lambda: eager_original(q, k, v), args.warmup, args.iterations)
    compiled_us = benchmark(
        lambda: compiled_original(q, k, v), args.warmup, args.iterations
    )
    custom_us = benchmark(lambda: custom(qkv), args.warmup, args.iterations)
    print(
        f"  attn_matmul max error={max_error:.3g}, eager={eager_us:.2f} us, "
        f"compiled={compiled_us:.2f} us, custom={custom_us:.2f} us, "
        f"speedup={eager_us / custom_us:.2f}x eager / "
        f"{compiled_us / custom_us:.2f}x compiled"
    )


def test_end_to_end(args, seq_len: int, channels: int, dtype: torch.dtype) -> None:
    torch._dynamo.reset()
    heads = channels // args.head_dim
    x = torch.randn(args.batch, seq_len, channels, device="cuda", dtype=dtype)
    original = OriginalLiteLA(
        channels, channels, heads=heads, dim=args.head_dim, qk_norm=False
    ).cuda().to(dtype).eval()
    custom_cls = (
        optimized_joint_litela
        if dtype == torch.bfloat16
        else JointLiteLA
    )
    custom = custom_cls(
        channels,
        channels,
        batch=args.batch,
        seq_len=seq_len,
        dtype=dtype,
        heads=heads,
        dim=args.head_dim,
        qk_norm=False,
    ).cuda().to(dtype).eval()
    custom.load_state_dict(original.state_dict())

    custom_op = CustomOpJointLiteLA(
        channels,
        channels,
        batch=args.batch,
        seq_len=seq_len,
        dtype=dtype,
        heads=heads,
        dim=args.head_dim,
        qk_norm=False,
    ).cuda().to(dtype).eval()
    custom_op.load_state_dict(original.state_dict())
    compiled_original = torch.compile(original, mode="max-autotune")
    compiled_custom_op = torch.compile(
        custom_op, mode="reduce-overhead", fullgraph=True
    )

    with torch.inference_mode():
        eager_expected = original(x).clone()
        compiled_expected = compiled_original(x).clone()
        actual = custom(x).clone()
        custom_op_output = compiled_custom_op(x).clone()
    torch.testing.assert_close(
        compiled_expected.float(), eager_expected.float(), rtol=2e-2, atol=2e-2
    )
    torch.testing.assert_close(actual.float(), eager_expected.float(), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(
        custom_op_output.float(), eager_expected.float(), rtol=2e-2, atol=2e-2
    )

    graph = CUDAGraphJointLiteLA(custom)
    graph_output = graph(x).clone()
    torch.testing.assert_close(
        graph_output.float(), eager_expected.float(), rtol=2e-2, atol=2e-2
    )

    eager_us = benchmark(lambda: original(x), args.joint_warmup, args.iterations)
    compiled_us = benchmark(
        lambda: compiled_original(x), args.joint_warmup, args.iterations
    )
    direct_us = benchmark(lambda: custom(x), args.joint_warmup, args.iterations)
    custom_op_us = benchmark(
        lambda: compiled_custom_op(x), args.joint_warmup, args.iterations
    )
    graph_us = benchmark(lambda: graph(x), args.joint_warmup, args.iterations)
    candidates = {
        type(custom).__name__: (direct_us, actual),
        "CustomOpJointLiteLA": (custom_op_us, custom_op_output),
        "CUDAGraphJointLiteLA": (graph_us, graph_output),
    }
    best_name, (best_us, best_output) = min(
        candidates.items(), key=lambda item: item[1][0]
    )
    max_error = (best_output.float() - eager_expected.float()).abs().max().item()
    print(
        f"  end_to_end max error={max_error:.3g}, eager={eager_us:.2f} us, "
        f"compiled={compiled_us:.2f} us, best={best_name} {best_us:.2f} us, "
        f"speedup={eager_us / best_us:.2f}x eager / "
        f"{compiled_us / best_us:.2f}x compiled"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[1024, 4096, 16384])
    parser.add_argument("--channel-sizes", type=int, nargs="+", default=[1152, 2240])
    parser.add_argument("--head-dim", type=int, default=32)
    parser.add_argument("--dtype", choices=DTYPES, default="bfloat16")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--joint-warmup", type=int, default=25)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("These tests require a CUDA-capable GPU.")
    if any(channels % args.head_dim for channels in args.channel_sizes):
        raise ValueError("every channel size must be divisible by head-dim")

    torch.manual_seed(0)
    dtype = DTYPES[args.dtype]
    for seq_len in args.seq_lens:
        for channels in args.channel_sizes:
            print(f"N={seq_len}, C={channels}, B={args.batch}, dtype={args.dtype}")
            test_attn_matmul(args, seq_len, channels, dtype)
            test_end_to_end(args, seq_len, channels, dtype)


if __name__ == "__main__":
    main()
