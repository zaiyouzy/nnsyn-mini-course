# Tested environment

This records the environment that produced the mini-course smoke run. It is evidence of what was tested, not a claim that every version is mandatory.

| Component | Tested value |
|---|---|
| Operating system | Windows 11 |
| Python | 3.11.9 |
| GPU | NVIDIA GeForce RTX 4070 Laptop GPU, 8 GB |
| nnsyn commit | `c3ba6fd8b32f62779f299f78b7d78a96b7fd7695` |
| nnunetv2 package | 2.5, installed from the nnsyn checkout |
| PyTorch | 2.13.0+cu130 |
| NumPy | 2.4.4 |
| Matplotlib | 3.11.1 |
| SimpleITK | 2.5.5 |
| batchgenerators | 0.25 |
| batchgeneratorsv2 | 0.3.5 |
| acvl-utils | 0.2.6 |
| dynamic-network-architectures | 0.4.2 |

The nnsyn package declares its dependency set in `pyproject.toml`. PyTorch is listed separately because the correct wheel depends on the operating system, driver, and available accelerator.