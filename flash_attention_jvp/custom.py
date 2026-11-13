"""FlashAttention forward/backward with a separate TileLang forward-mode rule."""

import math

import tilelang
import tilelang.language as T
import torch
from flash_attn.flash_attn_interface import _flash_attn_backward, _flash_attn_forward
from torch.autograd.function import once_differentiable


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
    block_m: int = 64,
    block_n: int = 64,
    threads: int = 128,
    num_stages: int = 2,
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


def _default_jvp_config(dim, keys, causal):
    # Recorded H20 defaults. Narrower key tiles keep padded D=256 within
    # shared-memory capacity; short scans do not need multistage prefetch.
    block_n, threads = (32, 256) if dim > 128 else (64, 128)
    stages = 1 if keys <= block_n or (not causal and keys >= 8192) else 2
    return dict(block_m=64, block_n=block_n, threads=threads, num_stages=stages)


def get_jvp_config(q, k, v, *, softmax_scale=None, causal=False):
    """Return deterministic default settings without compiling or tuning."""
    _validate_inputs(q, k, v, softmax_scale)
    return _default_jvp_config(q.shape[-1], k.shape[1], causal)


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
        kernel = _attention_jvp_kernel(
            q.shape[0], q.shape[1], k.shape[1], q.shape[2], k.shape[2], q.shape[3],
            "float16" if q.dtype == torch.float16 else "bfloat16", scale, causal,
            block_m, block_n, threads, num_stages,
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
    block_m=64, block_n=None, threads=None, num_stages=None,
):
    """Standalone JVP using cached forward results, without a forward launch.

    q/tq: [B, Nq, Hq, D]; k/tk/v/tv: [B, Nk, Hkv, D].
    out: [B, Nq, Hq, D]; lse: [B, Hq, Nq], float32, natural logarithms.
    out/lse must come from these same primals, scale and causal setting.
    A None tangent denotes zero. The result has q's dtype and out's shape.
    Default scheduling is (64, 64, 128, 2), with one stage for short scans and
    non-causal Nk >= 8192. Head dimensions above 128 use (64, 32, 256).
    Explicit schedule arguments override the corresponding defaults.
    """
    scale = _validate_inputs(q, k, v, softmax_scale)
    if out.shape != q.shape or out.dtype != q.dtype or out.device != q.device:
        raise ValueError("out must have q's shape, device, and dtype")
    if lse.shape != (q.shape[0], q.shape[2], q.shape[1]) or lse.dtype != torch.float32 or lse.device != q.device:
        raise ValueError("lse must be float32 [B, Hq, Nq] on q's device")
    defaults = _default_jvp_config(q.shape[-1], k.shape[1], causal)
    block_m = defaults["block_m"] if block_m is None else block_m
    block_n = defaults["block_n"] if block_n is None else block_n
    if threads is None:
        threads = 128 if block_m == 16 else defaults["threads"]
    if num_stages is None:
        num_stages = 1 if k.shape[1] <= block_n or (not causal and k.shape[1] >= 8192) else 2
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
    def forward(q, k, v, softmax_scale, causal):
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
        q, k, v, scale, causal = inputs
        out, lse = output
        ctx.softmax_scale = q.shape[-1] ** -0.5 if scale is None else float(scale)
        ctx.causal = causal
        ctx.set_materialize_grads(False)
        ctx.mark_non_differentiable(lse)
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.save_for_forward(q, k, v, out, lse)

    @staticmethod
    @once_differentiable
    def backward(ctx, dout, dlse):
        if dout is None:
            return (None,) * 5
        q, k, v, out, lse = ctx.saved_tensors
        dq, dk, dv = (
            torch.empty_like(x, memory_format=torch.contiguous_format) for x in (q, k, v)
        )
        _flash_attn_backward(
            dout.contiguous(), q, k, v, out, lse, dq, dk, dv,
            dropout_p=0.0, softmax_scale=ctx.softmax_scale, causal=ctx.causal,
            window_size_left=-1, window_size_right=-1, softcap=0.0,
            alibi_slopes=None, deterministic=False,
        )
        return dq, dk, dv, None, None

    @staticmethod
    def jvp(ctx, tq, tk, tv, tscale, tcausal):
        q, k, v, out, lse = ctx.saved_tensors
        tangent = flash_attention_jvp(
            q, k, v, tq, tk, tv, out, lse,
            softmax_scale=ctx.softmax_scale, causal=ctx.causal,
        )
        return tangent, None


def flash_attention(q, k, v, *, softmax_scale=None, causal=False, return_lse=False):
    """Attention with native forward/backward and custom forward-mode AD.

    Return out by default, or (out, lse) when return_lse=True. The natural-log
    LSE is a non-differentiable auxiliary output for standalone JVP calls.
    Supports torch.func.jvp and torch.autograd.forward_ad. Dropout is fixed at
    zero. Only first-order derivatives are implemented; vmap is not registered.
    """
    _validate_inputs(q, k, v, softmax_scale)
    out, lse = FlashAttentionFunction.apply(
        q.contiguous(), k.contiguous(), v.contiguous(), softmax_scale, bool(causal)
    )
    return (out, lse) if return_lse else out
