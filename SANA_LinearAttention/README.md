# SANA Linear Attention

This module provides the PyTorch LiteLA reference implementation and canonical
TileLang implementations of the complete SANA LiteLA path. `custom.py` contains
all retained kernels, modules, wrappers, and shape dispatch.

## Optimization Scope

Future kernel optimization, correctness validation, and performance benchmarks
target the following six SANA LiteLA input shapes. Module inputs have shape
`[B,N,C]`, with `B=1`, head dimension `D=32`, and `bfloat16` data; attention
kernels internally consume packed QKV or the grouped specialized path. `C=1152`
corresponds to SANA-0.6B, and `C=2240` corresponds to SANA-1.6B.

| Sequence length (`N`) | Channels (`C`) | Heads (`C / 32`) |
| ---: | ---: | ---: |
| 1024 | 1152 | 36 |
| 1024 | 2240 | 70 |
| 4096 | 1152 | 36 |
| 4096 | 2240 | 70 |
| 16384 | 1152 | 36 |
| 16384 | 2240 | 70 |

## Optimization Strategy

### 1. `attn_matmul`

The attention kernel uses the linear-attention identity
`O = (V ReLU(K)^T) ReLU(Q) / (sum(ReLU(K))^T ReLU(Q) + eps)`, avoiding an
`N x N` attention-score matrix. A producer kernel streams over sequence tiles
and accumulates the per-head `V ReLU(K)^T` matrix (`VK`) and `sum(ReLU(K))`
vector (`KSum`) in FP32 accumulators. The output kernel applies ReLU to Q,
multiplies it by `VK`, computes the normalization denominator, then casts the
result to the requested output dtype.

Long sequences are split into independently scheduled producer CTAs to improve
GPU occupancy. For `N >= 2048`, partial `VK` and `KSum` values are reduced
before the output pass; shorter sequences keep the partials and combine them
inside that pass to avoid an extra kernel launch. Producer tile width and thread
count, and output sequence tile width and thread count, are selected by
TileLang autotuning. The split count is dispatched from sequence length, batch,
and head count.

### 2. End-To-End LiteLA

`JointLiteLA` replaces the framework QKV and output projections with tiled
TileLang GEMMs, uses two-stage pipelining and transposed weights for reuse, and
fuses bias into the output projection. It reuses the optimized attention path
above. For exactly `B=1, N=1024, C=1152, H=36, D=32` in bf16, a specialized
path computes grouped K/V projections while producing attention partials, then
fuses Q projection, partial reduction, attention output, and normalization in
the Q/output CTA. This avoids materializing full Q, K, and V tensors and is
followed by the fused-bias output projection. Its producer uses 16 full 64-token
splits, groups two heads, and uses `block_k=32` with 128 threads; the Q/output
CTA uses `(head_group, token_tile, k_tile, threads)=(2,128,64,256)`. Other shapes
use the general `JointLiteLA` path. `CustomOpJointLiteLA` exposes the pipeline
as one operation to `torch.compile`; `CUDAGraphJointLiteLA` captures its
fixed-shape execution for replay.

## Canonical APIs

- `LiteLA`: packed-QKV TileLang attention with the framework output projection.
- `JointLiteLA`: TileLang QKV projection, attention, and bias-fused output projection.
- `GroupedFull64QOutLiteLA`: specialized retained implementation for exactly
  `B=1, N=1024, C=1152, H=36, D=32`, bf16, and `qk_norm=False`.
- `optimized_joint_litela`: constructs `GroupedFull64QOutLiteLA` for that exact
  shape and `JointLiteLA` otherwise.
- `CustomOpJointLiteLA` and `CUDAGraphJointLiteLA`: compile and fixed-shape CUDA
  Graph integration for the canonical joint pipeline.
- `qkv_projection` and `qkv_attn_matmul`: standalone kernel factories.

## Validation And Benchmark

Run from the repository root:

```bash
python SANA_LinearAttention/test.py
python SANA_LinearAttention/test.py --seq-lens 1024 --channel-sizes 1152
```

The combined test runs both attention-matmul and end-to-end comparisons for
each requested shape. By default both use `qk_norm=False`, `rotary_emb=None`,
`bfloat16`, batch 1, and head dimension 32, with numerical tolerance
`rtol=atol=2e-2`, GPU warm-up, and synchronization. Set `--dtype` to test another
supported dtype, `--batch` to change batch size, `--warmup` for attention warm-up,
or `--joint-warmup` for end-to-end warm-up.

Recorded on an NVIDIA RTX 6000D with PyTorch 2.8.0+cu128, CUDA 12.8, and
TileLang 0.1.13. Each row uses 100 timed iterations after 100 warm-up iterations
for attention, or 25 warm-up iterations end-to-end. The eager original is run
directly without `torch.compile`; the compiled original uses
`torch.compile(mode="max-autotune")`.

### Attention Matmul

The `attn_matmul` section compares eager and compiled original `attn_matmul`
with `qkv_attn_matmul`. It excludes QKV and output projections.

| Sequence length | Channels | Original eager (us) | Original compile (us) | Custom (us) | Speedup vs eager | Speedup vs compile |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 1152 | 70.69 | 65.87 | 24.81 | 2.85x | 2.66x |
| 1024 | 2240 | 87.96 | 70.17 | 39.27 | 2.24x | 1.79x |
| 4096 | 1152 | 158.19 | 80.32 | 49.71 | 3.18x | 1.62x |
| 4096 | 2240 | 300.46 | 189.86 | 89.79 | 3.35x | 2.11x |
| 16384 | 1152 | 707.93 | 466.07 | 187.31 | 3.78x | 2.49x |
| 16384 | 2240 | 1402.66 | 1122.26 | 371.06 | 3.78x | 3.02x |

Maximum absolute error against eager original was `0.00108`.

### End To End

The end-to-end section compares eager and compiled original LiteLA with the
fastest measured retained implementation among optimized direct execution,
custom-op execution, and CUDA Graph replay.

Although `attn_matmul` has standalone speedups, it accounts for only a modest
share of end-to-end runtime. The full path also includes QKV and output
projections and their associated kernel launches, so accelerating attention
matmul alone yields a smaller overall speedup.

| Sequence length | Channels | Best implementation | Original eager (us) | Original compile (us) | Best (us) | Speedup vs eager | Speedup vs compile |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1024 | 1152 | `CUDAGraphJointLiteLA` | 168.69 | 92.96 | 87.46 | 1.93x | 1.06x |
| 1024 | 2240 | `JointLiteLA` | 426.17 | 347.58 | 333.46 | 1.28x | 1.04x |
| 4096 | 1152 | `JointLiteLA` | 366.98 | 356.16 | 346.03 | 1.06x | 1.03x |
| 4096 | 2240 | `CUDAGraphJointLiteLA` | 1341.86 | 1260.50 | 1239.14 | 1.08x | 1.02x |
| 16384 | 1152 | `CUDAGraphJointLiteLA` | 1544.64 | 1390.34 | 1308.50 | 1.18x | 1.06x |
| 16384 | 2240 | `JointLiteLA` | 5446.02 | 5004.17 | 4849.72 | 1.12x | 1.03x |

Maximum absolute error against eager original was `0.000488`.
