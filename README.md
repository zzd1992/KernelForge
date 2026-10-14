# KernelForge

Custom GPU kernels for neural network operators and modules. The repository is updated periodically as new kernels and optimizations are added. Kernels are currently implemented primarily with [TileLang](https://github.com/tile-ai/tilelang).

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

## Results

| Module | Average Speedup | Core Optimization |
| :--- | ---: | :--- |
| **[RMSNorm](https://arxiv.org/pdf/1910.07467)** | **5.25x** | Two-pass scan to One-pass scan |

*Measured against PyTorch eager inference with `torch.inference_mode()`. See the module README for benchmark details.*
