# Tested environments

The full-data baseline and the optional smoke check ran in different environments. They also produced different self-configured plans, so their hardware and patch sizes should not be mixed.

## Full-data training and validation

This environment produced the 175-case, 300-epoch result reported in the course.

| Component | Tested value |
|---|---|
| System | Alliance Trillium compute node, Linux |
| Python | 3.11.5 |
| GPU | NVIDIA H100 80 GB HBM3 |
| NVIDIA driver | 580.173.02 |
| PyTorch | 2.13.0+computecanada |
| CUDA build reported by PyTorch | 13.2 |
| nnsyn commit | `c3ba6fd8b32f62779f299f78b7d78a96b7fd7695` |
| nnunetv2 package | 2.5, editable install from the pinned nnsyn checkout |
| MONAI | 1.5.2 |
| SimpleITK | 2.5.5 |
| batchgenerators | 0.25.1+computecanada |
| dynamic-network-architectures | 0.3.1 Alliance wheel |
| Data | 175 SynthRAD2025 Task 1 abdomen MRI–CT pairs |
| Plan | `3d_fullres`; patch `48 × 192 × 224`; batch size 2; spacing `3 × 1 × 1 mm` |

The non-tracking baseline still encounters nnsyn's optional AIM module during recursive trainer discovery. `full_run/compat/aim.py` provides only the import name needed by that discovery step; it raises an error if an AIM-tracking trainer is actually selected. Install the real `aim` package for experiment tracking.

`hiddenlayer` was not installed. nnsyn therefore skipped the optional network-graph export, but training and validation continued.

## Local smoke check

This environment produced the bounded three-case execution trace. It did not produce the main course result.

| Component | Tested value |
|---|---|
| Operating system | Windows 11 |
| Python | 3.11.9 |
| GPU | NVIDIA GeForce RTX 4070 Laptop GPU, 8 GB |
| PyTorch | 2.13.0+cu130 |
| CUDA build reported by PyTorch | 13.0 |
| nnsyn commit | `c3ba6fd8b32f62779f299f78b7d78a96b7fd7695` |
| nnunetv2 package | 2.5, editable install from the pinned nnsyn checkout |
| NumPy | 2.4.4 |
| Matplotlib | 3.11.1 |
| SimpleITK | 2.5.5 |
| batchgenerators | 0.25 |
| batchgeneratorsv2 | 0.3.5 |
| acvl-utils | 0.2.6 |
| dynamic-network-architectures | 0.4.2 |
| Data | Three staged SynthRAD2025 Task 1 abdomen cases |
| Plan | `3d_fullres`; patch `56 × 160 × 224`; batch size 2; spacing `3 × 1 × 1 mm` |

PyTorch is listed explicitly because its correct package depends on the operating system, driver, and available accelerator. The nnsyn checkout declares the remaining dependency set in `pyproject.toml`.
