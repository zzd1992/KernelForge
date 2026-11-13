"""Framework-native dense attention and JVP references in [B, S, H, D]."""

import torch


def attention(q, k, v, *, softmax_scale=None, causal=False):
    """Dense softmax attention with bottom-right causal alignment and GQA.

    Low-precision inputs use FP32 intermediates, then cast the output back.
    Float64 inputs stay float64 for independent derivative/finite-difference tests.
    """
    dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    qh, kh, vh = (x.to(dtype).transpose(1, 2) for x in (q, k, v))
    groups = q.shape[2] // k.shape[2]
    if groups > 1:
        kh, vh = (x.repeat_interleave(groups, dim=1) for x in (kh, vh))
    scale = q.shape[-1] ** -0.5 if softmax_scale is None else softmax_scale
    scores = (qh @ kh.transpose(-1, -2)) * scale
    if causal:
        nq, nk = q.shape[1], k.shape[1]
        rows = torch.arange(nq, device=q.device)[:, None]
        cols = torch.arange(nk, device=q.device)[None, :]
        valid = cols <= rows + nk - nq
        scores = scores.masked_fill(~valid, -torch.inf)
        # Native FlashAttention returns zero for entirely masked query rows.
        scores = torch.where(valid.any(dim=-1, keepdim=True), scores, 0.0)
    probabilities = scores.softmax(dim=-1)
    if causal:
        probabilities = probabilities.masked_fill(~valid, 0.0)
    return (probabilities @ vh).transpose(1, 2).to(q.dtype)


def attention_jvp(q, k, v, tq, tk, tv, *, softmax_scale=None, causal=False):
    """Return (out, tangent) by differentiating the dense PyTorch baseline."""
    tangents = tuple(torch.zeros_like(p) if t is None else t for p, t in zip((q, k, v), (tq, tk, tv)))
    return torch.func.jvp(
        lambda q, k, v: attention(q, k, v, softmax_scale=softmax_scale, causal=causal),
        (q, k, v), tangents,
    )


@torch.no_grad()
def attention_jvp_from_forward(q, k, v, tq, tk, tv, out, lse, *, softmax_scale=None, causal=False):
    """Dense standalone JVP with the same cached O/LSE as the custom kernel.

    This FP32 baseline materializes P and dS and is used for equivalent-work
    standalone benchmarks. Correctness tests independently use torch.func.jvp.
    """
    primals = (q, k, v)
    tangents = tuple(torch.zeros_like(p) if t is None else t for p, t in zip(primals, (tq, tk, tv)))
    qh, kh, vh, tqh, tkh, tvh = (x.float().transpose(1, 2) for x in (*primals, *tangents))
    groups = q.shape[2] // k.shape[2]
    if groups > 1:
        kh, vh, tkh, tvh = (x.repeat_interleave(groups, dim=1) for x in (kh, vh, tkh, tvh))
    scale = q.shape[-1] ** -0.5 if softmax_scale is None else softmax_scale
    probabilities = ((qh @ kh.transpose(-1, -2)) * scale - lse[..., None]).exp()
    if causal:
        nq, nk = q.shape[1], k.shape[1]
        rows = torch.arange(nq, device=q.device)[:, None]
        cols = torch.arange(nk, device=q.device)[None, :]
        probabilities = probabilities.masked_fill(cols > rows + nk - nq, 0.0)
    tscores = (tqh @ kh.transpose(-1, -2) + qh @ tkh.transpose(-1, -2)) * scale
    weighted = probabilities * tscores
    result = probabilities @ tvh + weighted @ vh - weighted.sum(-1, keepdim=True) * out.float().transpose(1, 2)
    return result.transpose(1, 2).to(q.dtype)
