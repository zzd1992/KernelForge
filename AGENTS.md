# KernelForge Agent Guidelines

Follow these repository conventions for every kernel implementation.

## Kernel Layout

- Put each kernel or optimized module in its own subdirectory.
- Make every kernel subdirectory importable from outside the directory. Include an `__init__.py` file and expose the intended public API from it.
- Keep one kernel or module family per subdirectory. Do not combine unrelated kernels in the same folder.

## Required Files

Every kernel subdirectory must contain these files:

- `README.md`: Module overview, core optimization strategy, and recorded benchmark results.
- `original.py`: The original, baseline, or framework-native operator/module implementation.
- `custom.py`: The custom GPU kernel or optimized module implementation.
- `test.py`: Correctness and performance tests comparing `custom.py` against `original.py`.

## Test Requirements

- Validate output correctness using suitable numerical tolerances for the operator's data type.
- Measure both implementations under equivalent inputs, shapes, data types, and device settings.
- Include GPU warm-up and synchronization when timing GPU work.
- Record benchmark configuration and results in the subdirectory's `README.md` clearly enough to reproduce the comparison.
