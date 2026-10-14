# RMSNorm

RMSNorm implementations in PyTorch and TileLang. The original TileLang implementation is based on the [official RMSNorm example](https://github.com/tile-ai/tilelang/blob/main/examples/norm/rms_norm.py).

Only the forward kernel is implemented currently. Backward propagation is not provided.

## Optimization Strategy

The custom TileLang kernel assigns one CUDA block to each `(batch, sequence)` row. Its primary optimization changes a two-scan RMSNorm into a one-scan input path.

### Two-Scan Baseline

RMSNorm requires the row-wide inverse root-mean-square before it can write any normalized element:

```text
inv_rms = rsqrt(sum(x[i] * x[i]) / dim + eps)
y[i] = x[i] * inv_rms * weight[i]
```

A conventional tiled implementation therefore processes a row in two phases. The first scan computes the sum of squares. Once the reduction produces `inv_rms`, the second scan loads the row again to normalize it and apply the weight. For rows that are not retained in registers across both phases, this adds a second read of the input tensor.

```text
Input row X
    |
    +-- scan 1: load X[i], accumulate X[i]^2 --> reduce --> inv_rms
    |
    +-- scan 2: load X[i] again, multiply by inv_rms and Weight[i] --> Y[i]
```

### One-Scan Custom Kernel

The custom kernel loads every assigned input element only once. Each thread converts its `bfloat16` input values to `float32`, stores them in its local register array, and accumulates the local sum of squares at the same time. After the block computes `inv_rms`, the thread reuses its retained values instead of reloading `X`.

```text
Input row X
    |
    +-- single load: X[i] --> x_local[i] and local_sum += x_local[i]^2
                                      |
                                      +--> block reduction --> inv_rms
                                                               |
    x_local[i] ------------------------------------------------+
    |
    +-- x_local[i] * inv_rms * Weight[i] --> cast to bfloat16 --> Y[i]
```

This trades a small, bounded amount of register storage for lower global-memory traffic. With 128 threads, a thread stores at most `ceil(dim / 128)` values; for the benchmark maximum of `dim=2048`, that is 16 `float32` values per thread.

### Thread and Block Layout

Each CUDA block owns one row `X[batch_idx, seq_idx, :]`, where `row = batch_idx * seq_len + seq_idx`. The block has 128 threads. Thread `t` owns strided columns `t`, `t + 128`, `t + 256`, and so on, so all threads together cover the row contiguously on each load iteration.

```text
Grid
  block row = 0                         block row = 1
  X[0, 0, :]                             X[0, 1, :]
  128 threads                            128 threads

One block for X[b, s, :]
  thread 0:   columns 0,   128, 256, ...
  thread 1:   columns 1,   129, 257, ...
  ...
  thread 127: columns 127, 255, 383, ...

  per thread: x_local[] + local sum of squares
       |
       +-- warp reduce --> one partial sum per warp in shared memory
                                  |
                                  +-- first warp reduces partial sums --> inv_rms
                                                                      |
  per thread: reuse x_local[] + load Weight[] --> Output[b, s, :]
```

The reduction is hierarchical: each warp reduces its local sums with `warp_reduce_sum`; lane 0 writes one partial sum per warp to shared memory; then the first warp reduces those partial sums. Block synchronizations ensure the shared partial sums and final `inv_rms` are visible before use.

## Benchmark

Run:

```bash
for seq_len in 256 1024 4096; do
  for dim in 128 512 2048; do
    python -m rmsnorm.test --batch 1 --seq-len "$seq_len" --dim "$dim" --warmup 100
  done
done
```

Measured on an NVIDIA RTX 6000D with `batch=1`, `bfloat16` inputs, 100 warm-up iterations, and 100 timed iterations. The PyTorch baseline runs eagerly under `torch.inference_mode()`.

| Sequence length | Dim | PyTorch | Original TileLang | Custom TileLang |
| ---: | ---: | ---: | ---: | ---: |
| 256 | 128 | 49.91 us | 28.11 us | 10.03 us |
| 256 | 512 | 42.31 us | 26.17 us | 9.20 us |
| 256 | 2048 | 62.71 us | 33.16 us | 13.06 us |
| 1024 | 128 | 48.17 us | 26.81 us | 9.88 us |
| 1024 | 512 | 74.50 us | 39.09 us | 17.29 us |
| 1024 | 2048 | 61.71 us | 28.96 us | 10.90 us |
| 4096 | 128 | 55.15 us | 31.05 us | 12.35 us |
| 4096 | 512 | 48.00 us | 28.47 us | 10.15 us |
| 4096 | 2048 | 113.10 us | 52.64 us | 12.86 us |

Across these nine shapes, the custom kernel is **5.25x faster than PyTorch** and **2.80x faster than the original TileLang implementation** on average. Each average is the arithmetic mean of the per-shape speedup ratios.

Results depend on GPU hardware, software versions, and tensor shapes.
