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

| Module | Baseline | Average Speedup | Core Optimization |
| :--- | :--- | ---: | :--- |
| **[RMSNorm](https://arxiv.org/pdf/1910.07467)** | PyTorch eager | **5.25x** | Two-pass scan to One-pass scan |
| **[SANA Linear Attention (`attn_matmul`)](SANA_LinearAttention/README.md)** | `torch.compile` | **2.28x** | Tiled linear-attention accumulation with sequence splitting |
| **[SANA Linear Attention (end-to-end)](SANA_LinearAttention/README.md)** | `torch.compile` | **1.04x** | Fused projection and attention kernels; CUDA Graph replay |
