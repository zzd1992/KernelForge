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
\dot O=P\dot V+HV-(H\mathbf{1})\odot O.
$$

Here, $\mathbf{1}$ is a column vector of ones. $H\mathbf{1}$ gives the row sums
of $H$, which are broadcast across the output feature dimension.

The kernel streams key tiles while retaining Q/dQ in shared memory. It performs
five Tensor Core GEMMs per key tile and accumulates scores, row corrections,
and output tangents in FP32. P/H tiles are cast to BF16 for Tensor Core
multiplication. It stores no full attention matrix in HBM.

Production calls use deterministic settings taken from the H20 tuning results:
the default `(block_m, block_n, threads, num_stages)` is `(64,64,128,2)`.
Non-causal key lengths of at least 8192 use one pipeline stage. Short scans
that fit in one key tile also use one stage. Head dimensions above 128 use
`(64,32,256)` tiles/threads to reduce shared-memory usage.

`T.Pipelined` schedules asynchronous KV prefetch and shared-memory buffering.
`custom.py` has no runtime autotuner, application-managed kernel/configuration
cache, cache keys, locks, or persistence callbacks. Compilation is handled by
the standard TileLang JIT; warm up a shape before CUDA Graph capture.
`get_jvp_config` reports default settings immediately, before any kernel launch.
Each explicitly supplied schedule argument overrides its default. The optional
offline schedule search is isolated in `test.py` (`--tune`).

## Usage

Inputs use `[B, sequence, heads, dim]` layout; LSE is FP32 `[B, heads, sequence]`.
`flash_attention` returns `out` by default; `return_lse=True` also returns the
non-differentiable auxiliary LSE for standalone JVP calls.

```python
import torch
from flash_attention_jvp import (
    flash_attention,
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
out, lse = flash_attention(q, k, v, causal=False, return_lse=True)
tangent = flash_attention_jvp(q, k, v, tq, tk, tv, out, lse, causal=False)
print(get_jvp_config(q, k, v, causal=False))
```

Cached O/LSE must match the same primals, scale, and mask used for JVP.

## Correctness and Timing

For all four sequence lengths, standalone JVP and the complete operator's
`torch.func.jvp` tangent passed comparison against dense PyTorch references
with `atol=3e-2, rtol=3e-2`. Old-fixed-versus-default JVP comparisons also passed.

The static-default implementation passes 47 correctness tests, including both
causal modes, FP16/BF16, head padding, GQA, native backward, both forward-mode
AD interfaces, explicit pipeline overrides, and graph replay with updated
tangent data. Package import and standalone `custom.py` import were verified
from outside the repository directory.

All timings below are in microseconds, measured using CUDA events with
synchronization and fixed-input CUDA Graph replay. Each implementation has
20 function warm-ups before capture and 20 replay warm-ups, followed by the
median of 5 batches of 100 replays. These results were recorded during the
earlier tuning runs, using the configurations now deployed as static defaults.
Compilation, tuning, and capture are excluded from the recorded timings.

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

### Default Configurations

| N | Query tile | Key tile | Threads | Pipeline stages |
| ---: | ---: | ---: | ---: | ---: |
| 1024 | 64 | 64 | 128 | 2 |
| 2048 | 64 | 64 | 128 | 2 |
| 4096 | 64 | 64 | 128 | 2 |
| 8192 | 64 | 64 | 128 | 1 |

The recorded `N=8192` winner uses one pipeline stage, while the shorter tested
sequences use two; these choices are now encoded in the default dispatch.

## Reproduction

Run from the repository root:

```bash
export CUDA_HOME=/usr/local/cuda-12.1
export PATH="$CUDA_HOME/bin:$PATH"

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
