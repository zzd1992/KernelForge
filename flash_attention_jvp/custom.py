"""FlashAttention forward/backward with a separate TileLang forward-mode rule."""

import hashlib
import inspect
import json
import math
import threading
from functools import lru_cache

import tilelang
import tilelang.language as T
import torch
from flash_attn.flash_attn_interface import _flash_attn_backward, _flash_attn_forward
from tilelang.autotuner import AutoTuner
from torch.autograd.function import once_differentiable


_TUNED_KERNELS = {}
_TUNE_LOCK = threading.Lock()


class _JVPAutoTuner(AutoTuner):
    def generate_cache_key(self, parameters, extra_parameters):
        # TileLang disables persistent caching for arbitrary supply/reference
        # callbacks. Ours have fixed semantics: the caller's read-only buffers
        # and the same 16x32 reference kernel. Give that controlled protocol an
        # explicit identity; cache compiled code/configuration, never tensors.
        identity = dict(
            protocol="kernelforge-jvp-real-inputs-v1",
            kernel=inspect.getsource(_attention_jvp_kernel.func),
            search=inspect.getsource(_jvp_configs),
            parameters=self._kernel_parameters,
            extra=extra_parameters,
            configs=self.configs,
            compile=hash(self.compile_args),
            profile=hash(self.profile_args),
            tilelang=tilelang.__version__,
            torch=torch.__version__,
            cuda=torch.version.cuda,
        )
        return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


@lru_cache(maxsize=None)
def _device_info(device):
    props = torch.cuda.get_device_properties(device)
    return props.name, props.major, props.minor, props.multi_processor_count


def _jvp_configs(queries, keys, dim, capability, causal=False):
    """Prune tiles by useful work, shared-memory footprint, and FP32 live state."""
    padded_dim = max(32, 1 << (dim - 1).bit_length())
    shared_limit = {(9, 0): 227 * 1024, (8, 0): 163 * 1024}.get(capability, 99 * 1024)
    query_tiles = (16, 32) if queries <= 32 else (32, 64) if queries <= 128 else (32, 64, 128)
    key_tiles = (32, 64) if keys <= 64 or padded_dim == 256 else (64, 128)
    # TileLang 0.1.15 cannot lower some long, CTA-dependent causal bounds at
    # depth three (tirx.Sub -> IntImm). Avoid those compiler-invalid trials.
    pipeline_stages = (1, 2) if causal else (1, 2, 3)
    configs = []
    for bm in query_tiles:
        for bn in key_tiles:
            # A 16-row/8-warp score layout cannot lower the row reduction in
            # TileLang 0.1.15. Keep 16-row schedules at four warps.
            for threads in ((128, 256) if bm >= 64 or bm >= 32 and bn == 128 else (128,)):
                # Leave register headroom for operands, reduction and address math.
                if (2 * bm * bn + bm * padded_dim) / threads > 128:
                    continue
                for stages in pipeline_stages:
                    if stages > 1 and keys <= bn or stages > 2 and keys <= 2 * bn:
                        continue
                    shared_bytes = 2 * (2 * bm * padded_dim + 4 * stages * bn * padded_dim + 2 * bm * bn)
                    if shared_bytes + 8192 > shared_limit:
                        continue
                    configs.append(dict(block_m=bm, block_n=bn, threads=threads, num_stages=stages))
    return configs


@tilelang.jit(out_idx=[-1], target="cuda")
def _attention_jvp_kernel(
    batch: int,
    queries: int,
    keys: int,
    heads: int,
    kv_heads: int,
    dim: int,
    dtype: str,
    scale: float,
    causal: bool,
    block_m: int = 32,
    block_n: int = 64,
    threads: int = 128,
    num_stages: int = 1,
):
    """Return only dO; O and natural-log LSE come from FlashAttention forward."""
    padded_dim = max(32, 1 << (dim - 1).bit_length())
    log2e = 1.4426950408889634
    group_size = heads // kv_heads

    @T.prim_func
    def main(
        Q: T.Tensor((batch, queries, heads, dim), dtype),
        K: T.Tensor((batch, keys, kv_heads, dim), dtype),
        V: T.Tensor((batch, keys, kv_heads, dim), dtype),
        TQ: T.Tensor((batch, queries, heads, dim), dtype),
        TK: T.Tensor((batch, keys, kv_heads, dim), dtype),
        TV: T.Tensor((batch, keys, kv_heads, dim), dtype),
        O: T.Tensor((batch, queries, heads, dim), dtype),  # noqa: E741 - mathematical output O
        LSE: T.Tensor((batch, heads, queries), "float32"),
        TO: T.Tensor((batch, queries, heads, dim), dtype),
    ):
        with T.Kernel(T.ceildiv(queries, block_m), heads, batch, threads=threads) as (
            query_tile, head, batch_idx
        ):
            q = T.alloc_shared((block_m, padded_dim), dtype)
            tq = T.alloc_shared((block_m, padded_dim), dtype)
            k = T.alloc_shared((block_n, padded_dim), dtype)
            tk = T.alloc_shared((block_n, padded_dim), dtype)
            v = T.alloc_shared((block_n, padded_dim), dtype)
            tv = T.alloc_shared((block_n, padded_dim), dtype)
            p_shared = T.alloc_shared((block_m, block_n), dtype)
            h_shared = T.alloc_shared((block_m, block_n), dtype)
            scores = T.alloc_fragment((block_m, block_n), "float32")
            tscores = T.alloc_fragment((block_m, block_n), "float32")
            tangent = T.alloc_fragment((block_m, padded_dim), "float32")
            row_lse = T.alloc_fragment((block_m,), "float32")
            row_sum = T.alloc_fragment((block_m,), "float32")
            correction = T.alloc_fragment((block_m,), "float32")

            kv_head = head // group_size
            for i, d in T.Parallel(block_m, padded_dim):
                token = query_tile * block_m + i
                q[i, d] = T.if_then_else(token < queries and d < dim, Q[batch_idx, token, head, d], 0)
                tq[i, d] = T.if_then_else(token < queries and d < dim, TQ[batch_idx, token, head, d], 0)
            for i in T.Parallel(block_m):
                token = query_tile * block_m + i
                # Empty causal rows have +inf LSE in FlashAttention. They are
                # explicitly masked below, avoiding any inf-inf subtraction.
                row_lse[i] = T.if_then_else(token < queries, LSE[batch_idx, head, token], 0)
            T.clear(tangent)
            T.clear(correction)

            # FlashAttention aligns a rectangular causal mask at the bottom right.
            key_stop = keys
            if causal:
                key_stop = T.min(keys, T.max(0, (query_tile + 1) * block_m + keys - queries))
            for key_tile in T.Pipelined(T.ceildiv(key_stop, block_n), num_stages=num_stages):
                for j, d in T.Parallel(block_n, padded_dim):
                    token = key_tile * block_n + j
                    k[j, d] = T.if_then_else(token < keys and d < dim, K[batch_idx, token, kv_head, d], 0)
                    tk[j, d] = T.if_then_else(token < keys and d < dim, TK[batch_idx, token, kv_head, d], 0)
                    v[j, d] = T.if_then_else(token < keys and d < dim, V[batch_idx, token, kv_head, d], 0)
                    tv[j, d] = T.if_then_else(token < keys and d < dim, TV[batch_idx, token, kv_head, d], 0)

                T.clear(scores)
                T.clear(tscores)
                T.gemm(q, k, scores, transpose_B=True)
                T.gemm(tq, k, tscores, transpose_B=True)
                T.gemm(q, tk, tscores, transpose_B=True)
                for i, j in T.Parallel(block_m, block_n):
                    query_idx = query_tile * block_m + i
                    key_idx = key_tile * block_n + j
                    valid = query_idx < queries and key_idx < keys
                    if causal:
                        valid = valid and key_idx <= query_idx + keys - queries
                    # Only logits use log2 units; score tangents stay in natural
                    # units, as in Eq. (3), so no log2e belongs in tscores.
                    scores[i, j] = T.if_then_else(
                        valid, T.exp2((scores[i, j] * scale - row_lse[i]) * log2e), 0
                    )
                    tscores[i, j] = scores[i, j] * tscores[i, j] * scale
                T.reduce_sum(tscores, row_sum, dim=1)
                for i in T.Parallel(block_m):
                    correction[i] += row_sum[i]
                T.copy(scores, p_shared)
                T.copy(tscores, h_shared)
                T.gemm(p_shared, tv, tangent)
                T.gemm(h_shared, v, tangent)

            for i, d in T.Parallel(block_m, padded_dim):
                token = query_tile * block_m + i
                if token < queries and d < dim:
                    TO[batch_idx, token, head, d] = (
                        tangent[i, d] - correction[i] * T.Cast("float32", O[batch_idx, token, head, d])
                    )

    return main


def _kernel_args(q, k, scale, causal):
    return (
        q.shape[0], q.shape[1], k.shape[1], q.shape[2], k.shape[2], q.shape[3],
        "float16" if q.dtype == torch.float16 else "bfloat16", scale, bool(causal),
    )


def _config_key(q, k, scale, causal):
    return (_kernel_args(q, k, scale, causal), q.device.index, _device_info(q.device.index))


def get_jvp_config(q, k, v, *, softmax_scale=None, causal=False):
    """Return the cached autotuned schedule, or None if this shape is not warm."""
    scale = _validate_inputs(q, k, v, softmax_scale)
    kernel = _TUNED_KERNELS.get(_config_key(q, k, scale, causal))
    return None if kernel is None else dict(kernel.config)


def _tune_jvp_kernel(inputs, scale, causal):
    """Use TileLang's persistent tuner with real inputs, not random LSE/O."""
    q, k = inputs[:2]
    args = _kernel_args(q, k, scale, causal)
    info = _device_info(q.device.index)
    configs = _jvp_configs(q.shape[1], k.shape[1], q.shape[3], (info[1], info[2]), causal)
    # Use a low-resource schedule as a validated reference on every device.
    # Full correctness against FP64 PyTorch is separately covered in test.py.
    reference_config = dict(block_m=16, block_n=32, threads=128, num_stages=1)
    reference = _attention_jvp_kernel(*args, **reference_config)
    torch.cuda.synchronize(q.device)
    tuner = _JVPAutoTuner(_attention_jvp_kernel.func, configs)
    tuner.set_kernel_parameters(
        (args, (("device", q.device.index), ("gpu", info))),
        _attention_jvp_kernel.signature.parameters,
    )
    def compile_candidate(**config):
        config.pop("__pass_configs__", None)
        return _attention_jvp_kernel(*args, **config)

    def elaborate_candidate(**config):
        config.pop("__pass_configs__", None)
        return _attention_jvp_kernel.get_tir(*args, **config)

    tuner.jit_compile = compile_candidate
    tuner.jit_elaborate = elaborate_candidate
    tuner.set_compile_args(out_idx=[-1], target="cuda", execution_backend="tvm_ffi")
    atol, rtol = (3e-3, 1e-2) if q.dtype == torch.float16 else (3e-2, 3e-2)
    tuner.set_profile_args(
        supply_prog=lambda _: inputs, ref_prog=reference,
        atol=atol, rtol=rtol, max_mismatched_ratio=0.0,
        cache_input_tensors=False, backend="cudagraph", warmup=5, rep=15,
    )
    result = tuner.run(warmup=5, rep=15)
    result.kernel.config = result.config
    result.kernel.latency = result.latency
    return result.kernel


@torch.library.custom_op("kernelforge::flash_attention_jvp", mutates_args=(), device_types="cuda")
def _launch_jvp(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    tq: torch.Tensor, tk: torch.Tensor, tv: torch.Tensor,
    out: torch.Tensor, lse: torch.Tensor,
    scale: float, causal: bool, block_m: int, block_n: int, threads: int, num_stages: int,
) -> torch.Tensor:
    # An opaque torch operator unwraps torch.func tracking tensors before the
    # external runtime accesses their storage. This also supplies a fake rule.
    with torch.cuda.device(q.device):
        if block_m == 0:
            key = _config_key(q, k, scale, causal)
            kernel = _TUNED_KERNELS.get(key)
            if kernel is None:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError("Warm up this JVP shape before CUDA Graph capture to populate its autotune cache")
                with _TUNE_LOCK:
                    kernel = _TUNED_KERNELS.get(key)
                    if kernel is None:
                        kernel = _tune_jvp_kernel((q, k, v, tq, tk, tv, out, lse), scale, causal)
                        _TUNED_KERNELS[key] = kernel
        else:
            kernel = _attention_jvp_kernel(
                *_kernel_args(q, k, scale, causal), block_m, block_n, threads, num_stages,
            )
        return kernel(q, k, v, tq, tk, tv, out, lse)


@_launch_jvp.register_fake
def _launch_jvp_fake(q, k, v, tq, tk, tv, out, lse, scale, causal, block_m, block_n, threads, num_stages):
    return torch.empty_like(out)


def _validate_inputs(q, k, v, softmax_scale):
    if any(x.ndim != 4 for x in (q, k, v)):
        raise ValueError("q, k, v must have shape [batch, sequence, heads, dim]")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("FlashAttention JVP supports float16 and bfloat16")
    if not q.is_cuda or any(x.device != q.device or x.dtype != q.dtype for x in (k, v)):
        raise ValueError("q, k, v must use the same CUDA device and dtype")
    b, nq, h, d = q.shape
    if min(b, nq, h, d, k.shape[1], k.shape[2]) <= 0:
        raise ValueError("batch, sequence lengths, heads, and head dimension must be positive")
    if k.shape != v.shape or k.shape[0] != b or k.shape[-1] != d:
        raise ValueError("k and v must match, with the same batch and head dimension as q")
    if h % k.shape[2] or d % 8 or d > 256:
        raise ValueError("query heads must be divisible by KV heads; dim must be a multiple of 8 <= 256")
    scale = d ** -0.5 if softmax_scale is None else float(softmax_scale)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("softmax_scale must be finite and positive")
    return scale


def _prepare_tangent(tangent, primal, name):
    if tangent is None:
        return torch.zeros_like(primal)
    if tangent.shape != primal.shape or tangent.device != primal.device or tangent.dtype != primal.dtype:
        raise ValueError(f"{name} must have the same shape, device, and dtype as its primal")
    return tangent.contiguous()


@torch.no_grad()
def flash_attention_jvp(
    q, k, v, tq, tk, tv, out, lse, *, softmax_scale=None, causal=False,
    block_m=None, block_n=None, threads=None, num_stages=None,
):
    """Standalone JVP using cached forward results, without a forward launch.

    q/tq: [B, Nq, Hq, D]; k/tk/v/tv: [B, Nk, Hkv, D].
    out: [B, Nq, Hq, D]; lse: [B, Hq, Nq], float32, natural logarithms.
    out/lse must come from these same primals, scale and causal setting.
    A None tangent denotes zero. The result has q's dtype and out's shape.
    Scheduling is autotuned on first use and cached. Setting any scheduling
    argument bypasses tuning; unspecified arguments then use (32, 64, 128, 1).
    """
    scale = _validate_inputs(q, k, v, softmax_scale)
    if out.shape != q.shape or out.dtype != q.dtype or out.device != q.device:
        raise ValueError("out must have q's shape, device, and dtype")
    if lse.shape != (q.shape[0], q.shape[2], q.shape[1]) or lse.dtype != torch.float32 or lse.device != q.device:
        raise ValueError("lse must be float32 [B, Hq, Nq] on q's device")
    if all(x is None for x in (block_m, block_n, threads, num_stages)):
        block_m, block_n, threads, num_stages = 0, 0, 0, 0
    else:
        block_m = 32 if block_m is None else block_m
        block_n = 64 if block_n is None else block_n
        threads = 128 if threads is None else threads
        num_stages = 1 if num_stages is None else num_stages
        if (block_m not in (16, 32, 64, 128) or block_n not in (32, 64, 128)
                or threads not in (128, 256) or num_stages not in (1, 2, 3)):
            raise ValueError("unsupported tile/pipeline configuration")
    q, k, v = (x.contiguous() for x in (q, k, v))
    tq, tk, tv = (_prepare_tangent(t, p, n) for t, p, n in zip((tq, tk, tv), (q, k, v), ("tq", "tk", "tv")))
    return _launch_jvp(
        q, k, v, tq, tk, tv, out.contiguous(), lse.contiguous(),
        scale, bool(causal), block_m, block_n, threads, num_stages,
    )


class FlashAttentionFunction(torch.autograd.Function):
    """Native FlashAttention VJP plus a TileLang JVP, with torch.func support."""

    @staticmethod
    def forward(q, k, v, softmax_scale, causal, deterministic):
        scale = _validate_inputs(q, k, v, softmax_scale)
        out, lse, _, _ = _flash_attn_forward(
            q, k, v, dropout_p=0.0, softmax_scale=scale, causal=causal,
            window_size_left=-1, window_size_right=-1, softcap=0.0,
            alibi_slopes=None, return_softmax=False,
        )
        # LSE is an auxiliary output so setup_context can save it without
        # running forward a second time (required by torch.func transforms).
        return out, lse

    @staticmethod
    def setup_context(ctx, inputs, output):
        q, k, v, scale, causal, deterministic = inputs
        out, lse = output
        ctx.softmax_scale = q.shape[-1] ** -0.5 if scale is None else float(scale)
        ctx.causal = causal
        ctx.deterministic = deterministic
        ctx.set_materialize_grads(False)
        ctx.mark_non_differentiable(lse)
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.save_for_forward(q, k, v, out, lse)

    @staticmethod
    @once_differentiable
    def backward(ctx, dout, dlse):
        if dout is None:
            return (None,) * 6
        q, k, v, out, lse = ctx.saved_tensors
        dq, dk, dv = (
            torch.empty_like(x, memory_format=torch.contiguous_format) for x in (q, k, v)
        )
        _flash_attn_backward(
            dout.contiguous(), q, k, v, out, lse, dq, dk, dv,
            dropout_p=0.0, softmax_scale=ctx.softmax_scale, causal=ctx.causal,
            window_size_left=-1, window_size_right=-1, softcap=0.0,
            alibi_slopes=None, deterministic=ctx.deterministic,
        )
        return dq, dk, dv, None, None, None

    @staticmethod
    def jvp(ctx, tq, tk, tv, tscale, tcausal, tdeterministic):
        q, k, v, out, lse = ctx.saved_tensors
        tangent = flash_attention_jvp(
            q, k, v, tq, tk, tv, out, lse,
            softmax_scale=ctx.softmax_scale, causal=ctx.causal,
        )
        return tangent, None


def flash_attention_forward(q, k, v, *, softmax_scale=None, causal=False, deterministic=False):
    """Return (out, lse), using the autograd operator with cached statistics."""
    _validate_inputs(q, k, v, softmax_scale)
    return FlashAttentionFunction.apply(
        q.contiguous(), k.contiguous(), v.contiguous(), softmax_scale, bool(causal), bool(deterministic)
    )


def flash_attention(q, k, v, *, softmax_scale=None, causal=False, deterministic=False):
    """Attention with native forward/backward and custom forward-mode AD.

    Supports torch.func.jvp and torch.autograd.forward_ad. Dropout is fixed at
    zero. Only first-order derivatives are implemented; vmap is not registered.
    """
    return flash_attention_forward(
        q, k, v, softmax_scale=softmax_scale, causal=causal, deterministic=deterministic
    )[0]
