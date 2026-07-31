# Reproduce the verified nnsyn walkthrough

This folder contains the code behind the real-data evidence in the **Inside nnsyn** mini-course. Each script corresponds to a specific step that was run locally and discussed in the course.

This is a teaching-scale execution check, not a performance benchmark.

## What is included

| File | What it does | Evidence it produced |
|---|---|---|
| `make_course_figures.py` | Reads one staged SynthRAD2025 case and nnsyn's preprocessed arrays. | Figures 01–05: paired volumes, normalization, a training patch, the inherited target slot, and alignment. |
| `run_nnsyn_smoke_training.py` | Constructs the nnsyn trainer directly and runs one epoch with two training iterations. | A complete data-loader → forward → loss → backward → validation → checkpoint path. |
| `run_smoke_inference.py` | Predicts the held-out case, restores CT Hounsfield units, applies the body mask, calculates masked MAE, and makes the four-panel comparison. | Figure 06 and the observed masked MAE of 249.9 HU. |
| `patches/nnsyn_course_windows.patch` | Makes local execution independent of Slurm signals and fixes two unused imports that break trainer discovery. | The course run can train and predict on Windows. |
| `patches/nnUNetTrainer_nnsyn_smoke.py` | Defines the bounded two-iteration course trainer. | Keeps the execution check short and clearly separate from real training. |

The smoke result is intentionally poor because the model received only two optimizer updates. It verifies execution, not image quality, convergence, generalization, or clinical suitability.

## Tested setup

The run was verified on Windows 11 with Python 3.11.9, an NVIDIA RTX 4070 Laptop GPU with 8 GB memory, and the package versions recorded in [`ENVIRONMENT.md`](ENVIRONMENT.md). The nnsyn source was pinned to commit `c3ba6fd8b32f62779f299f78b7d78a96b7fd7695`.

CPU execution is supported by the smoke-training wrapper but is substantially slower. Inference was tested with CUDA.

## Clone and install nnsyn

Run these commands from the root of this mini-course repository:

```powershell
git clone https://github.com/aehrc/nnsyn.git nnsyn
git -C nnsyn checkout c3ba6fd8b32f62779f299f78b7d78a96b7fd7695

py -3.11 -m venv .venv-nnsyn
. .\.venv-nnsyn\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -e .\nnsyn
python -m pip check
```

The PyTorch command reproduces the tested NVIDIA setup. Select the platform-appropriate PyTorch build if CUDA 13.0 is not suitable for your machine.

## Apply the course compatibility changes

The pinned nnsyn commit assumes Slurm-style signals in one local-training path, and two unused imports prevent recursive trainer discovery during inference. Apply the documented patch and install the bounded trainer:

```powershell
git -C nnsyn apply ..\code\patches\nnsyn_course_windows.patch

$trainerDestination = ".\nnsyn\nnunetv2\training\nnUNetTrainer\variants\nnsyn\nnUNetTrainer_nnsyn_smoke.py"
Copy-Item .\code\patches\nnUNetTrainer_nnsyn_smoke.py $trainerDestination
```

These changes are compatibility and teaching controls. They do not change the network architecture, image loss, or prediction reconstruction.

## Data and preprocessing

Obtain SynthRAD2025 Task 1 training data through the [official data page](https://synthrad2025.grand-challenge.org/data/) and follow its access and licence conditions. Medical images are not redistributed here.

The walkthrough used three abdomen cases: `1ABA033`, `1ABB062`, and `1ABB118`. Stage them as:

```text
dataset/
└── nnsyn_origin/
    └── Task1_AB_3cases/
        ├── INPUT_IMAGES/
        │   ├── 1ABA033_0000.mha
        │   ├── 1ABB062_0000.mha
        │   └── 1ABB118_0000.mha
        ├── TARGET_IMAGES/
        │   ├── 1ABA033_0000.mha
        │   ├── 1ABB062_0000.mha
        │   └── 1ABB118_0000.mha
        └── MASKS/
            ├── 1ABA033.mha
            ├── 1ABB062.mha
            └── 1ABB118.mha
```

`INPUT_IMAGES` contains MRI, `TARGET_IMAGES` contains paired CT, and `MASKS` identifies the valid body region.

Set the paths:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
. .\code\setup_course_env.ps1
```

Then plan and preprocess Dataset501:

```powershell
nnsyn_plan_and_preprocess -d 501 `
    --preprocessing_input MR `
    --preprocessing_target CT `
    --dataset_name Task1_AB_3cases `
    --use_mask
```

In the verified run, nnsyn selected the `3d_fullres` configuration, a `56 × 160 × 224` patch, batch size 2, and `3 × 1 × 1 mm` target spacing.

## Smoke training

First construct and inspect the trainer without updating weights:

```powershell
python .\code\run_nnsyn_smoke_training.py `
    --dataset 501 --fold 0 --device cuda --check-only
```

Then run the bounded execution check:

```powershell
python .\code\run_nnsyn_smoke_training.py `
    --dataset 501 --fold 0 --device cuda
```

The script creates deterministic leave-one-out folds for a dataset with fewer than five cases. Fold 0 trains on two cases and holds out `1ABA033`.

Observed on the tested machine:

```text
epochs:                 1
training iterations:    2
validation iterations:  1
train loss:             1.8487
validation loss:        0.8441
elapsed time:           54.62 s
```

These numbers describe one execution check. They are not expected performance.

## Inference, HU restoration, and evaluation

```powershell
python .\code\run_smoke_inference.py --case 1ABA033 --workers 1
```

The wrapper calls `nnUNetv2_predict` directly because the current `nnsyn_predict` wrapper parses several worker and inference options but does not forward them. It then reuses nnsyn's own `revert_normalisation` helper.

Observed output:

```text
masked MAE: 249.9 HU
prediction: nnsyn_workspace/course_predictions/.../1ABA033.mha
figure:     mini_course/figures/fig06_smoke_synthetic_ct.png
```

The visible sliding-window seams are expected from a nearly untrained model. They explain the inference mechanism; they are not evidence of the quality of a trained nnsyn model.

## Rebuild the evidence figures

```powershell
python .\code\make_course_figures.py
```

This regenerates figures 01–05 from the staged case and Dataset501 arrays. Figure 06 is regenerated by the inference script.

## Claim boundaries

This walkthrough supports the following statements:

- The selected public data can be staged and preprocessed by nnsyn.
- The planned 3D network, data loader, loss, optimizer, validation, and checkpoint path execute together.
- A held-out MRI can be predicted, restored to HU, masked, compared with its paired CT, and visualized.
- Several portability and wrapper assumptions can be located and documented from source code.

It does **not** support a claim about trained image quality, challenge performance, loss-variant superiority, generalization, or clinical use.

For the full framework, cite and consult the [aehrc/nnsyn repository](https://github.com/aehrc/nnsyn). For the dataset, cite the SynthRAD2025 data publication and follow the official terms.