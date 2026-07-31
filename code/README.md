# Reproduce the verified nnsyn walkthrough

This code produced the real-data evidence in the **Inside nnsyn** mini-course. It is a teaching-scale execution check, not a performance benchmark.

## Included evidence

| File | Purpose |
|---|---|
| `make_course_figures.py` | Rebuilds Figures 01–05 from one staged SynthRAD2025 case and nnsyn arrays. |
| `run_nnsyn_smoke_training.py` | Runs one epoch with two training iterations and one validation iteration. |
| `run_smoke_inference.py` | Predicts the held-out case, restores HU, applies the mask, calculates MAE, and builds Figure 06. |
| `patches/` | Adds the bounded trainer and fixes local Windows execution and trainer discovery. |

The model received only two optimizer updates. The run verifies execution, not image quality, convergence, generalization, or clinical suitability.

## Tested setup

The run used Windows 11, Python 3.11.9, an RTX 4070 Laptop GPU with 8 GB, and nnsyn commit `c3ba6fd8b32f62779f299f78b7d78a96b7fd7695`. Exact package versions are in [`ENVIRONMENT.md`](ENVIRONMENT.md).

## Install

Run from the mini-course repository root:

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

Use the platform-appropriate PyTorch build if CUDA 13.0 is unsuitable.

Apply the course compatibility patch and copy the bounded trainer:

```powershell
git -C nnsyn apply ..\code\patches\nnsyn_course_windows.patch
$trainerDestination = ".\nnsyn\nnunetv2\training\nnUNetTrainer\variants\nnsyn\nnUNetTrainer_nnsyn_smoke.py"
Copy-Item .\code\patches\nnUNetTrainer_nnsyn_smoke.py $trainerDestination
```

The patch disables Slurm-only signals during local runs and removes two unused imports that stop inference-time trainer discovery. It does not change the network, loss, or reconstruction.

## Data and preprocessing

Obtain SynthRAD2025 Task 1 training data through the [official data page](https://synthrad2025.grand-challenge.org/data/) and follow its licence. Images are not redistributed here.

The walkthrough used abdomen cases `1ABA033`, `1ABB062`, and `1ABB118`:

```text
dataset/nnsyn_origin/Task1_AB_3cases/
├── INPUT_IMAGES/<case>_0000.mha    # MRI
├── TARGET_IMAGES/<case>_0000.mha   # paired CT
└── MASKS/<case>.mha                # body mask
```

Set the paths and preprocess Dataset501:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
. .\code\setup_course_env.ps1

nnsyn_plan_and_preprocess -d 501 `
    --preprocessing_input MR `
    --preprocessing_target CT `
    --dataset_name Task1_AB_3cases `
    --use_mask
```

The verified plan used `3d_fullres`, patch `56 × 160 × 224`, batch size 2, and spacing `3 × 1 × 1 mm`.

## Smoke training

Inspect the trainer without updating weights, then run it:

```powershell
python .\code\run_nnsyn_smoke_training.py --dataset 501 --fold 0 --device cuda --check-only
python .\code\run_nnsyn_smoke_training.py --dataset 501 --fold 0 --device cuda
```

For fewer than five cases, the wrapper creates deterministic leave-one-out folds. Fold 0 trains on two cases and holds out `1ABA033`.

Observed execution: train loss `1.8487`, validation loss `0.8441`, elapsed time `54.62 s`. These are trace values, not expected performance.

## Inference and figures

```powershell
python .\code\run_smoke_inference.py --case 1ABA033 --workers 1
python .\code\make_course_figures.py
```

Inference calls `nnUNetv2_predict` directly because the current `nnsyn_predict` wrapper does not forward several parsed worker and inference options. It then reuses nnsyn's `revert_normalisation` helper.

Observed masked MAE was `249.9 HU`. The visible patch seams come from a nearly untrained model and illustrate sliding-window inference; they are not trained-model performance.

## Claim boundary

This walkthrough verifies that the selected public data can pass through planning, preprocessing, loading, prediction, loss, backpropagation, validation, checkpointing, inference, HU restoration, masking, evaluation, and visualization. It does not establish challenge performance, superiority of one loss, broader generalization, or clinical use.

Consult and cite [aehrc/nnsyn](https://github.com/aehrc/nnsyn) and the SynthRAD2025 data publication.