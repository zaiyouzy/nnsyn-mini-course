# Reproduce the verified nnsyn walkthrough

This folder contains the code used for the **Inside nnsyn** mini-course. The main course result comes from a 300-epoch, full-data training run and evaluation of all 35 cases in held-out fold 0. Separate smoke utilities remain available as a quick local execution check.

## Included evidence

| File | Purpose |
|---|---|
| `make_course_figures.py` | Rebuilds Figures 01–05 from one staged SynthRAD2025 case and nnsyn arrays. |
| `run_nnsyn_smoke_training.py` | Runs one epoch with two training iterations and one validation iteration. |
| `run_smoke_inference.py` | Runs the optional three-case smoke inference check. |
| `evaluate_trained_validation.py` | Restores HU, applies body masks, evaluates all 35 held-out cases, and builds the representative result figure. |
| `results/` | Per-case masked MAE and the aggregate summary from the 300-epoch fold-0 model. |
| `patches/` | Adds the bounded trainer and fixes local Windows execution and trainer discovery. |

The smoke model receives only two optimizer updates and is not the result shown in the course. The reported course result uses the best checkpoint from our separate 300-epoch training run.

## Full-data result reported in the course

| Item | Value |
|---|---|
| Data | 175 paired SynthRAD2025 Task 1 abdomen cases |
| Split | Fold 0: 140 training, 35 held-out validation |
| Model | 3D PlainConvUNet, masked MSE |
| Training | 300 epochs, best validation checkpoint |
| Mean / median masked MAE | 105.0 / 102.0 HU |
| Range | 67.4–189.1 HU |
| Representative case | 1ABA101, 102.0 HU |
| Device | NVIDIA H100 80 GB GPU |

The representative case was selected as the case nearest the cohort median. The numbers are one-fold validation results, not challenge-test or external-site results. See [`results/validation_mae.csv`](results/validation_mae.csv) and [`results/validation_summary.json`](results/validation_summary.json).
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

The patch disables Slurm-only signals during local runs and removes two unused imports that stop inference-time trainer discovery. It does not change the network, loss, or reconstruction. It was generated against nnsyn commit `c3ba6fd8b32f62779f299f78b7d78a96b7fd7695`. If Windows line endings cause whitespace-only conflicts, retry with `git -C nnsyn apply --ignore-whitespace ..\code\patches\nnsyn_course_windows.patch`; do not use that option to hide substantive conflicts on another commit.

## Data and preprocessing

Obtain SynthRAD2025 Task 1 training data through the [official data page](https://synthrad2025.grand-challenge.org/data/) and follow its license. Images are not redistributed here.

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

## Optional local smoke training

Inspect the trainer without updating weights, then run it:

```powershell
python .\code\run_nnsyn_smoke_training.py --dataset 501 --fold 0 --device cuda --check-only
python .\code\run_nnsyn_smoke_training.py --dataset 501 --fold 0 --device cuda
```

For fewer than five cases, the wrapper creates deterministic leave-one-out folds. Fold 0 trains on two cases and holds out `1ABA033`.

Observed execution: train loss `1.8487`, validation loss `0.8441`, elapsed time `54.62 s`. These are trace values, not expected performance.

## Optional smoke inference

```powershell
python .\code\run_smoke_inference.py --case 1ABA033 --workers 1
```

Inference calls `nnUNetv2_predict` directly because the current `nnsyn_predict` wrapper does not forward several parsed worker and inference options. It then reuses nnsyn's `revert_normalisation` helper.

The smoke output is retained only as an execution diagnostic and is not used as the course performance result.

## Rebuild the evidence figures

```powershell
python .\code\make_course_figures.py
```

This script recreates Figures 01–05 from the public teaching subset. The published Figure 06 comes from the 300-epoch validation evaluation.

## Claim boundary

The smoke utilities verify pipeline connectivity. The full-data run additionally shows that the masked-MSE PlainConvUNet can learn an MRI-to-CT mapping and complete inference, HU restoration, masking, and evaluation across 35 held-out cases. It does not establish challenge-test performance, five-fold or ensemble performance, superiority of one loss, external-site generalization, or clinical use.

Consult and cite [aehrc/nnsyn](https://github.com/aehrc/nnsyn) and the SynthRAD2025 data publication.

For the full AI-assistance statement, human verification notes, and source acknowledgements, see the [repository README](../README.md).
