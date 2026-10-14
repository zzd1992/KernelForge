# KernelForge

Custom GPU kernels for neural network operators and modules. The repository is updated periodically as new kernels and optimizations are added.

Kernels are currently implemented primarily with [TileLang](https://github.com/tile-ai/tilelang).

## Results

| Module | Average speedup | Core optimization |
| --- | ---: | --- |
| [RMSNorm](rmsnorm/) | 5.71x vs. PyTorch | Single-pass scan that retains loaded values for normalization and weighting |

## Layout

Each kernel lives in its own importable subdirectory:

```text
kernel_name/
  __init__.py
  original.py  # Baseline implementation
  custom.py    # Optimized GPU implementation
  test.py      # Correctness and performance tests
```

Tests compare `custom.py` with `original.py` for correctness and speed using identical inputs, shapes, data types, and device settings. GPU benchmarks must include warm-up iterations and synchronization.
