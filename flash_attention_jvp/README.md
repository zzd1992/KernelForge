# FlashAttention JVP

An independent TileLang JVP kernel wrapped in `torch.autograd.Function`.
Forward and backward use FlashAttention's native functions; JVP uses the cached
forward output and LSE and returns only the output tangent.

## Benchmark Configuration

All recorded results use the following configuration:

| Setting | Value |
| --- | --- |
| Input/output dtype | BF16 |
| Batch | `B=1` |
| Query/KV heads | `Hq=Hkv=16` |
| Head dimension | `D=64` |
| Causal | `False` |
| Query/key sequence lengths | `Nq=Nk=N`, with `N=1024/2048/4096/8192` |
| Softmax scale / dropout | `0.125` / `0` |
| Input/tangent distribution | Random normal, seed 2026; all three tangents nonzero |
| GPU | NVIDIA H20, SM90 |
| Software | PyTorch 2.5.1+cu121, CUDA toolkit 12.1.105, TileLang 0.1.15, FlashAttention 2.7.4.post1 |

## Kernel Strategy

For each query/key tile, reconstruct probabilities from native forward's
natural-log LSE, then compute

$$
\dot S=\gamma(\dot QK^\top+Q\dot K^\top),\qquad
H=P\odot\dot S,\qquad
\dot O=P\dot V+HV-\operatorname{rowsum}(H)\odot O.
$$

The kernel streams key tiles while retaining Q/dQ in shared memory. It performs
five Tensor Core GEMMs per key tile and accumulates scores, row corrections,
and output tangents in FP32. P/H tiles are cast to BF16 for Tensor Core
multiplication. It stores no full attention matrix in HBM.

Tile sizes, thread count, and pipeline depth are jointly autotuned, with
shared-memory and FP32 fragment pressure pruning. `T.Pipelined` schedules
asynchronous KV prefetch and shared-memory buffering. Candidates are checked
using the caller's real inputs and cached O/LSE before CUDA Graph profiling.
Selected code/configurations are cached in memory and on disk. Warm up a shape
before CUDA Graph capture.

## Usage

Inputs use `[B, sequence, heads, dim]` layout; LSE is FP32 `[B, heads, sequence]`.

```python
import torch
from flash_attention_jvp import (
    flash_attention,
    flash_attention_forward,
    flash_attention_jvp,
    get_jvp_config,
)

q, k, v = (
    torch.randn(1, 1024, 16, 64, device="cuda", dtype=torch.bfloat16)
    for _ in range(3)
)
tq, tk, tv = (torch.randn_like(x) for x in (q, k, v))

# Complete operator: native forward, independent TileLang JVP.
out, tangent = torch.func.jvp(flash_attention, (q, k, v), (tq, tk, tv))

# Standalone JVP with cached native forward results.
out, lse = flash_attention_forward(q, k, v, causal=False)
tangent = flash_attention_jvp(q, k, v, tq, tk, tv, out, lse, causal=False)
print(get_jvp_config(q, k, v, causal=False))
```

Cached O/LSE must match the same primals, scale, and mask used for JVP.

## Correctness and Timing

For all four sequence lengths, standalone JVP and the complete operator's
`torch.func.jvp` tangent passed comparison against dense PyTorch references
with `atol=3e-2, rtol=3e-2`. Fixed-versus-tuned JVP comparisons also passed.

All timings below are in microseconds, measured using CUDA events with
synchronization and fixed-input CUDA Graph replay. Each implementation has
20 function warm-ups before capture and 20 replay warm-ups, followed by the
median of 5 batches of 100 replays. Compilation, autotuning, and capture are
excluded. The two comparison tables were measured in separate invocations.

### Against Dense PyTorch

Standalone implementations share identical cached native O/LSE; their timings
exclude forward. Full-path timings compare dense `torch.func.jvp` attention
against `torch.func.jvp(flash_attention, ...)`, including native forward and
the independent JVP launch. Dense references use FP32 intermediates and BF16
inputs/outputs; TF32 is disabled.

| N | Dense JVP | TileLang JVP | Speedup | Dense forward+JVP | Custom forward+JVP | Speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 994.95 | 103.73 | 9.59x | 1391.43 | 172.47 | 8.07x |
| 2048 | 3473.94 | 390.94 | 8.89x | 5260.54 | 649.41 | 8.10x |
| 4096 | 13352.86 | 1331.86 | 10.03x | 19997.77 | 2206.18 | 9.06x |
| 8192 | 53420.11 | 5011.79 | 10.66x | 80571.94 | 8449.72 | 9.54x |

The arithmetic mean standalone speedup across these four sequence lengths is
**9.79x**.

### Selected Configurations

| N | Query tile | Key tile | Threads | Pipeline stages |
| ---: | ---: | ---: | ---: | ---: |
| 1024 | 64 | 64 | 128 | 2 |
| 2048 | 64 | 64 | 128 | 2 |
| 4096 | 64 | 64 | 128 | 2 |
| 8192 | 64 | 64 | 128 | 1 |

The `N=8192` search selects one pipeline stage, while the shorter tested
sequences select two. Increasing pipeline depth is not uniformly beneficial
as sequence length grows.

## Reproduction

Run from the repository root:

```bash
export CUDA_HOME=/usr/local/cuda-12.1
export PATH="$CUDA_HOME/bin:$PATH"
export TILELANG_AUTO_TUNING_CPU_COUNTS=4

python -m flash_attention_jvp.test --benchmark --cuda-graph \
  --dtype bfloat16 --batch 1 --heads 16 --dim 64 \
  --seq-lens 1024 2048 4096 8192 --warmup 20 --iterations 100 --repeats 5

python -m flash_attention_jvp.test --compare-schedules \
  --dtype bfloat16 --batch 1 --heads 16 --dim 64 \
  --seq-lens 1024 2048 4096 8192 --warmup 20 --iterations 100 --repeats 5
```

Both commands use `causal=False` by default and check correctness before timing.

## References

- [Mathematical derivation](flash-attention-jvp-derivation.md).
- [Ryu1845's Triton example](https://gist.github.com/Ryu1845/904a9b2c1fd58ca0b5d14119a5452eea).
- [NVlabs/rcm Triton implementation](https://github.com/NVlabs/rcm/blob/main/rcm/utils/flash_attention_jvp_triton.py).
