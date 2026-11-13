"""Run correctness with pytest, and reproducible GPU benchmarks as a module."""

import argparse
import statistics
from importlib.metadata import version
from unittest.mock import patch

import pytest
import torch
from flash_attn import flash_attn_func
from tilelang import env as tilelang_env

from .custom import (
    _attention_jvp_kernel,
    _config_key,
    _JVPAutoTuner,
    _jvp_configs,
    _TUNED_KERNELS,
    flash_attention,
    flash_attention_jvp,
    get_jvp_config,
)
from .original import attention, attention_jvp, attention_jvp_from_forward


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
DTYPES = (torch.float16, torch.bfloat16)
# B, Nq, Nk, Hq, Hkv, D, causal, scale. Includes partial tiles, head padding,
# bottom-right rectangular masks, fully masked rows, and GQA/MQA.
CASES = (
    (1, 7, 11, 2, 2, 8, False, None),
    (2, 33, 65, 4, 4, 24, False, 0.3),
    (1, 129, 131, 3, 3, 64, False, None),
    (1, 65, 65, 2, 2, 64, True, None),
    (1, 19, 71, 4, 2, 96, True, None),
    (1, 71, 19, 4, 1, 32, True, None),
    (1, 37, 91, 6, 2, 128, False, 0.2),
    (1, 9, 13, 2, 1, 256, False, None),
    (1, 1, 1, 2, 2, 64, True, None),
    (1, 17, 29, 2, 2, 16, False, 0.05),
    (1, 23, 37, 2, 2, 80, False, 0.15),
    (1, 999, 515, 2, 2, 64, False, None),
)


def make_inputs(case, dtype, *, noncontiguous=False):
    b, nq, nk, h, hk, d, _, _ = case
    torch.manual_seed(2026)
    shapes = ((b, nq, h, d), (b, nk, hk, d), (b, nk, hk, d))
    inputs = []
    for shape in (*shapes, *shapes):
        if noncontiguous:
            # Include a non-unit last stride; neither primals nor tangents may
            # be assumed to share strides in the standalone API.
            expanded = (*shape[:-1], shape[-1] * 2)
            x = torch.randn(expanded, device="cuda", dtype=dtype)[..., ::2]
        else:
            x = torch.randn(shape, device="cuda", dtype=dtype)
        inputs.append(x)
    return tuple(inputs)


def tolerance(dtype):
    return {torch.float16: dict(atol=3e-3, rtol=1e-2), torch.bfloat16: dict(atol=3e-2, rtol=3e-2)}[dtype]


def double_reference(inputs):
    # Use CPU FP64 for a genuinely independent oracle, including on GPUs whose
    # FP64 GEMM accuracy differs from CPU (observed on the test H20).
    return tuple(x.detach().cpu().double() for x in inputs)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("case", CASES)
def test_standalone_jvp(case, dtype):
    q, k, v, tq, tk, tv = make_inputs(case, dtype)
    kwargs = dict(causal=case[-2], softmax_scale=case[-1])
    out, lse = flash_attention(q, k, v, return_lse=True, **kwargs)
    # Assert that requesting only dO never invokes the native forward again.
    with patch("flash_attention_jvp.custom._flash_attn_forward", side_effect=AssertionError("extra forward")):
        actual = flash_attention_jvp(q, k, v, tq, tk, tv, out, lse, **kwargs)
    ref_out, ref_tangent = attention_jvp(*double_reference((q, k, v, tq, tk, tv)), **kwargs)
    torch.testing.assert_close(out.cpu().double(), ref_out, **tolerance(dtype))
    torch.testing.assert_close(actual.cpu().double(), ref_tangent, **tolerance(dtype))
    dense = attention_jvp_from_forward(q, k, v, tq, tk, tv, out, lse, **kwargs)
    torch.testing.assert_close(actual, dense, **tolerance(dtype))
    assert actual.shape == out.shape and actual.dtype == dtype and torch.isfinite(actual).all()
    if kwargs["causal"] and q.shape[1] > k.shape[1]:
        assert torch.count_nonzero(actual[:, :q.shape[1] - k.shape[1]]) == 0


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("causal", (False, True))
def test_autograd_interfaces(dtype, causal):
    case = (1, 67, 41, 4, 2, 64, causal, 0.17)
    inputs = make_inputs(case, dtype, noncontiguous=True)
    q, k, v, tq, tk, tv = inputs
    kwargs = dict(causal=causal, softmax_scale=case[-1], deterministic=True)
    def fn(q, k, v):
        return flash_attention(q, k, v, **kwargs)

    def fn_with_lse(q, k, v):
        return flash_attention(q, k, v, return_lse=True, **kwargs)

    ref_out, ref_tangent = attention_jvp(*double_reference(inputs), causal=causal, softmax_scale=case[-1])

    out, tangent = torch.func.jvp(fn, (q, k, v), (tq, tk, tv))
    torch.testing.assert_close(out.cpu().double(), ref_out, **tolerance(dtype))
    torch.testing.assert_close(tangent.cpu().double(), ref_tangent, **tolerance(dtype))
    (stats_out, lse), (stats_tangent, lse_tangent) = torch.func.jvp(fn_with_lse, (q, k, v), (tq, tk, tv))
    torch.testing.assert_close(stats_out, out, atol=0, rtol=0)
    torch.testing.assert_close(stats_tangent, tangent, atol=0, rtol=0)
    assert lse.shape == (q.shape[0], q.shape[2], q.shape[1]) and lse.dtype == torch.float32
    assert torch.count_nonzero(lse_tangent) == 0
    with torch.autograd.forward_ad.dual_level():
        duals = tuple(torch.autograd.forward_ad.make_dual(p, t) for p, t in zip((q, k, v), (tq, tk, tv)))
        dual_out = fn(*duals)
        primal, forward_tangent = torch.autograd.forward_ad.unpack_dual(dual_out)
        torch.testing.assert_close(primal, out, atol=0, rtol=0)
        torch.testing.assert_close(forward_tangent, tangent, atol=0, rtol=0)
        dual_stats_out, dual_lse = fn_with_lse(*duals)
        stats_primal, stats_forward_tangent = torch.autograd.forward_ad.unpack_dual(dual_stats_out)
        lse_primal, lse_forward_tangent = torch.autograd.forward_ad.unpack_dual(dual_lse)
        torch.testing.assert_close(stats_primal, primal, atol=0, rtol=0)
        torch.testing.assert_close(stats_forward_tangent, forward_tangent, atol=0, rtol=0)
        torch.testing.assert_close(lse_primal, lse, atol=0, rtol=0)
        assert lse_forward_tangent is None

    primals = tuple(x.detach().requires_grad_() for x in (q, k, v))
    cotangent = torch.randn_like(out)
    actual_out = fn(*primals)
    actual_grads = torch.autograd.grad(actual_out, primals, cotangent)
    stats_out, lse = fn_with_lse(*primals)
    assert not lse.requires_grad
    stats_grads = torch.autograd.grad(stats_out, primals, cotangent)
    native_out = flash_attn_func(*primals, **kwargs)
    native_grads = torch.autograd.grad(native_out, primals, cotangent)
    torch.testing.assert_close(actual_out, native_out, atol=0, rtol=0)
    torch.testing.assert_close(stats_out, native_out, atol=0, rtol=0)
    double_primals = tuple(x.requires_grad_() for x in double_reference(primals))
    reference_out = attention(*double_primals, causal=causal, softmax_scale=case[-1])
    reference_grads = torch.autograd.grad(reference_out, double_primals, cotangent.cpu().double())
    for actual, stats, native, reference in zip(actual_grads, stats_grads, native_grads, reference_grads):
        torch.testing.assert_close(actual, native, **tolerance(dtype))
        torch.testing.assert_close(stats, native, **tolerance(dtype))
        torch.testing.assert_close(actual.cpu().double(), reference, **tolerance(dtype))


@pytest.mark.parametrize("dtype", DTYPES)
def test_missing_and_value_only_tangents(dtype):
    case = (1, 35, 49, 2, 2, 64, False, None)
    q, k, v, tq, tk, tv = make_inputs(case, dtype)
    out, lse = flash_attention(q, k, v, return_lse=True)
    for tangents in ((None, None, None), (None, None, tv), (tq, None, None), (None, tk, None)):
        actual = flash_attention_jvp(q, k, v, *tangents, out, lse)
        ref_tangents = tuple(None if x is None else x.cpu().double() for x in tangents)
        _, reference = attention_jvp(*double_reference((q, k, v)), *ref_tangents)
        torch.testing.assert_close(actual.cpu().double(), reference, **tolerance(dtype))
    with torch.autograd.forward_ad.dual_level():
        dual_v = torch.autograd.forward_ad.make_dual(v, tv)
        _, actual = torch.autograd.forward_ad.unpack_dual(flash_attention(q, k, dual_v))
    expected = flash_attention_jvp(q, k, v, None, None, tv, out, lse)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", DTYPES)
def test_uniform_score_shift_and_tile_sizes(dtype):
    case = (1, 39, 73, 2, 2, 64, False, None)
    q, k, v, tq, tk, tv = make_inputs(case, dtype)
    tk = torch.ones_like(k) * 0.25  # Q dK^T is constant across keys.
    out, lse = flash_attention(q, k, v, return_lse=True)
    shifted = flash_attention_jvp(q, k, v, None, tk, None, out, lse)
    torch.testing.assert_close(shifted, torch.zeros_like(shifted), **tolerance(dtype))
    _, reference = attention_jvp(*double_reference((q, k, v, tq, tk, tv)))
    for bm, bn, threads in ((16, 32, 128), (32, 64, 128), (64, 128, 256)):
        actual = flash_attention_jvp(q, k, v, tq, tk, tv, out, lse, block_m=bm, block_n=bn, threads=threads)
        torch.testing.assert_close(actual.cpu().double(), reference, **tolerance(dtype))


def test_finite_difference_and_duality():
    case = (1, 23, 31, 4, 2, 32, True, None)
    inputs = make_inputs(case, torch.float16)
    q, k, v, tq, tk, tv = inputs
    out, lse = flash_attention(q, k, v, causal=True, return_lse=True)
    actual = flash_attention_jvp(*inputs, out, lse, causal=True)
    primals, tangents = double_reference(inputs[:3]), double_reference(inputs[3:])
    epsilon = 1e-4
    plus = attention(*(p + epsilon * t for p, t in zip(primals, tangents)), causal=True)
    minus = attention(*(p - epsilon * t for p, t in zip(primals, tangents)), causal=True)
    torch.testing.assert_close(actual.cpu().double(), (plus - minus) / (2 * epsilon), **tolerance(q.dtype))
    primals = tuple(x.requires_grad_() for x in inputs[:3])
    cotangent = torch.randn_like(out)
    grads = torch.autograd.grad(flash_attention(*primals, causal=True), primals, cotangent)
    lhs = (actual.float() * cotangent.float()).sum()
    rhs = sum((g.float() * t.float()).sum() for g, t in zip(grads, inputs[3:]))
    torch.testing.assert_close(lhs, rhs, atol=2e-2, rtol=1e-2)


def test_input_validation():
    case = (1, 9, 13, 2, 2, 64, False, None)
    q, k, v, tq, tk, tv = make_inputs(case, torch.float16)
    out, lse = flash_attention(q, k, v, return_lse=True)
    with pytest.raises(ValueError, match="lse"):
        flash_attention_jvp(q, k, v, tq, tk, tv, out, lse.half())
    with pytest.raises(ValueError, match="tq"):
        flash_attention_jvp(q, k, v, tq[:, :-1], tk, tv, out, lse)
    with pytest.raises(ValueError, match="finite"):
        flash_attention(q, k, v, softmax_scale=float("nan"))
    with pytest.raises(ValueError, match="positive"):
        flash_attention(q, k, v, softmax_scale=0.0)
    with pytest.raises(ValueError, match="float16"):
        flash_attention(q.float(), k.float(), v.float())
    with pytest.raises(ValueError, match="pipeline"):
        flash_attention_jvp(q, k, v, tq, tk, tv, out, lse, num_stages=4)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("causal", (False, True))
@pytest.mark.parametrize("stages", (2, 3))
def test_pipeline_tails_and_empty_rows(dtype, causal, stages):
    inputs = make_inputs((1, 193, 97, 4, 2, 64, causal, None), dtype)
    out, lse = flash_attention(*inputs[:3], causal=causal, return_lse=True)
    actual = flash_attention_jvp(
        *inputs, out, lse, causal=causal, block_m=32, block_n=32, threads=128, num_stages=stages,
    )
    _, reference = attention_jvp(*double_reference(inputs), causal=causal)
    torch.testing.assert_close(actual.cpu().double(), reference, **tolerance(dtype))
    if causal:
        assert torch.count_nonzero(actual[:, :96]) == 0


def test_autotune_cache_and_graph_replay():
    inputs = make_inputs((1, 67, 131, 4, 2, 64, True, None), torch.float16)
    out, lse = flash_attention(*inputs[:3], causal=True, return_lse=True)
    actual = flash_attention_jvp(*inputs, out, lse, causal=True)
    config = get_jvp_config(*inputs[:3], causal=True)
    assert config is not None and set(config) == {"block_m", "block_n", "threads", "num_stages"}
    graph = torch.cuda.CUDAGraph()
    with patch("flash_attention_jvp.custom._tune_jvp_kernel", side_effect=AssertionError("repeated tuning")):
        with torch.cuda.graph(graph):
            captured = flash_attention_jvp(*inputs, out, lse, causal=True)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(captured, actual, atol=0, rtol=0)
        inputs[5].add_(0.03125)
        graph.replay()
        torch.cuda.synchronize()
        expected = flash_attention_jvp(*inputs, out, lse, causal=True)
        torch.testing.assert_close(captured, expected, atol=0, rtol=0)
    assert get_jvp_config(*inputs[:3], causal=True) == config
    if tilelang_env.is_cache_enabled() and not tilelang_env.is_autotune_cache_disabled():
        # Model a fresh process: remove both in-memory layers and forbid new
        # profiling. The saved code/configuration must load from disk correctly.
        _TUNED_KERNELS.pop(_config_key(inputs[0], inputs[1], 64 ** -0.5, True))
        with patch.object(_JVPAutoTuner, "_memory_cache", {}):
            with patch.object(_JVPAutoTuner, "_benchmark_target", side_effect=AssertionError("cache miss")):
                restored = flash_attention_jvp(*inputs, out, lse, causal=True)
        torch.testing.assert_close(restored, expected, atol=0, rtol=0)
        assert get_jvp_config(*inputs[:3], causal=True) == config


def benchmark_cuda(fn, warmup, iterations, repeats, *, cuda_graph=False):
    """Median of CUDA-event batches, including warm-up and synchronization."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    if cuda_graph:
        # Capture both implementations under identical settings. Outputs stay
        # alive until timing completes; all inputs remain fixed during replay.
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _captured_output = fn()
        fn = graph.replay
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000 / iterations)
    return statistics.median(times)


def run_benchmarks(args):
    torch.backends.cuda.matmul.allow_tf32 = False
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    print(f"GPU: {torch.cuda.get_device_name()}; torch {torch.__version__}; CUDA {torch.version.cuda}; "
          f"tilelang {version('tilelang')}; flash-attn {version('flash-attn')}")
    print(f"dtype={args.dtype}, B={args.batch}, Hq=Hkv={args.heads}, D={args.dim}, "
          f"causal={args.causal}, scale=D**-0.5, seed=2026, TF32=False, "
          f"warmup={args.warmup}, iterations={args.iterations}, repeats={args.repeats}, "
          f"cuda_graph={args.cuda_graph}")
    print("| N | Dense JVP us | TileLang JVP us | Speedup | Dense forward+JVP us | Custom forward+JVP us | Speedup |")
    print("| ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for n in args.seq_lens:
        inputs = make_inputs((args.batch, n, n, args.heads, args.heads, args.dim, args.causal, None), dtype)
        q, k, v, tq, tk, tv = inputs
        kwargs = dict(causal=args.causal)
        out, lse = flash_attention(q, k, v, return_lse=True, **kwargs)
        def standalone_custom():
            return flash_attention_jvp(*inputs, out, lse, **kwargs)

        def standalone_dense():
            return attention_jvp_from_forward(*inputs, out, lse, **kwargs)

        def full_custom():
            return torch.func.jvp(
                lambda q, k, v: flash_attention(q, k, v, **kwargs), (q, k, v), (tq, tk, tv)
            )

        def full_dense():
            return attention_jvp(*inputs, **kwargs)

        for actual, reference in ((standalone_custom(), standalone_dense()), (full_custom()[1], full_dense()[1])):
            torch.testing.assert_close(actual, reference, **tolerance(dtype))
        functions = (standalone_dense, standalone_custom, full_dense, full_custom)
        times = [benchmark_cuda(fn, args.warmup, args.iterations, args.repeats, cuda_graph=args.cuda_graph)
                 for fn in functions]
        dense, custom, dense_full, custom_full = times
        print(f"| {n} | {dense:.2f} | {custom:.2f} | {dense/custom:.2f}x | "
              f"{dense_full:.2f} | {custom_full:.2f} | {dense_full/custom_full:.2f}x |", flush=True)
        print(f"Selected N={n}: {get_jvp_config(q, k, v, **kwargs)}", flush=True)


def run_schedule_sweep(args):
    """Validate and graph-time all resource-pruned schedules on real O/LSE."""
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    capability = torch.cuda.get_device_capability()
    for n in args.seq_lens:
        inputs = make_inputs((args.batch, n, n, args.heads, args.heads, args.dim, args.causal, None), dtype)
        out, lse = flash_attention(*inputs[:3], causal=args.causal, return_lse=True)
        reference = attention_jvp_from_forward(*inputs, out, lse, causal=args.causal)
        configs = _jvp_configs(n, n, args.dim, capability, args.causal)
        print(f"N={n}, dtype={args.dtype}, D={args.dim}, causal={args.causal}, candidates={len(configs)}")
        times = []
        for config in configs:
            kernel = _attention_jvp_kernel(
                args.batch, n, n, args.heads, args.heads, args.dim, args.dtype,
                args.dim ** -0.5, args.causal, **config,
            )
            torch.testing.assert_close(kernel(*inputs, out, lse), reference, **tolerance(dtype))

            def launch():
                return kernel(*inputs, out, lse)

            us = benchmark_cuda(launch, args.warmup, args.iterations, args.repeats, cuda_graph=True)
            times.append((us, config))
            print(f"{config}: {us:.2f} us", flush=True)
        print(f"Best: {min(times, key=lambda result: result[0])}", flush=True)


def run_schedule_comparison(args):
    """Compare the former fixed schedule and autotuning on identical GPU work."""
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    old_config = dict(block_m=32, block_n=64, threads=128, num_stages=1)
    print(f"GPU={torch.cuda.get_device_name()}, dtype={args.dtype}, B={args.batch}, H={args.heads}, "
          f"D={args.dim}, causal={args.causal}, graph=True, warmup={args.warmup}, "
          f"iterations={args.iterations}, repeats={args.repeats}")
    print("| N | Fixed JVP us | Tuned JVP us | Speedup | Fixed forward+JVP us | Tuned forward+JVP us | Speedup |")
    print("| ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for n in args.seq_lens:
        inputs = make_inputs((args.batch, n, n, args.heads, args.heads, args.dim, args.causal, None), dtype)
        out, lse = flash_attention(*inputs[:3], causal=args.causal, return_lse=True)

        def fixed_jvp():
            return flash_attention_jvp(*inputs, out, lse, causal=args.causal, **old_config)

        def tuned_jvp():
            return flash_attention_jvp(*inputs, out, lse, causal=args.causal)

        def full(config):
            output, stats = flash_attention(*inputs[:3], causal=args.causal, return_lse=True)
            tangent = flash_attention_jvp(*inputs, output, stats, causal=args.causal, **config)
            return output, tangent

        torch.testing.assert_close(tuned_jvp(), fixed_jvp(), **tolerance(dtype))
        times = [benchmark_cuda(fn, args.warmup, args.iterations, args.repeats, cuda_graph=True)
                 for fn in (fixed_jvp, tuned_jvp, lambda: full(old_config), lambda: full({}))]
        fixed, tuned, fixed_full, tuned_full = times
        print(f"| {n} | {fixed:.2f} | {tuned:.2f} | {fixed/tuned:.2f}x | "
              f"{fixed_full:.2f} | {tuned_full:.2f} | {fixed_full/tuned_full:.2f}x |", flush=True)
        print(f"Selected N={n}: {get_jvp_config(*inputs[:3], causal=args.causal)}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", action="store_true", help="run benchmarks instead of correctness tests")
    parser.add_argument("--tune", action="store_true", help="validate and print every candidate schedule's latency")
    parser.add_argument("--compare-schedules", action="store_true", help="graph-time fixed versus tuned schedules")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--seq-lens", nargs="+", type=int, default=(128, 512, 1024, 2048))
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--cuda-graph", action="store_true", help="time fixed-input CUDA Graph replay")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    if min(args.batch, args.heads, args.dim, args.iterations, args.repeats, *args.seq_lens) <= 0 or args.warmup < 0:
        parser.error("sizes/iterations/repeats must be positive and warmup nonnegative")
    if args.tune:
        run_schedule_sweep(args)
    elif args.compare_schedules:
        run_schedule_comparison(args)
    elif args.benchmark:
        run_benchmarks(args)
    else:
        torch.set_num_threads(4)
        raise SystemExit(pytest.main([__file__, "-q"]))


if __name__ == "__main__":
    main()
