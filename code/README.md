# Reproduce the nnsyn mini-course experiments

This folder has two separate paths:

1. **Full-data baseline:** the 175-case, 300-epoch experiment reported in the course.
2. **Local smoke check:** a three-case run that only checks whether an installation can preprocess, train, save a checkpoint, and predict.

The smoke check is optional and is not the source of the course result.

## What is included

| Path | Purpose |
|---|---|
| `full_run/run_full_baseline.py` | Runs data staging, preprocessing, 300-epoch training, and 35-case evaluation as explicit stages. |
| `full_run/prepare_synthrad_ab.py` | Converts the official `<case>/{mr,ct,mask}.mha` layout into the flat folders expected by nnsyn. |
| `full_run/nnUNetTrainer_nnsyn_loss_masked_300epochs.py` | Exact 300-epoch trainer subclass used by the baseline. |
| `full_run/trillium_*.sbatch` | Slurm jobs matching the Alliance Trillium run. |
| `evaluate_trained_validation.py` | Restores HU, applies masks, evaluates all 35 held-out cases, and makes the representative result figure. |
| `results/` | Per-case masked MAE and aggregate statistics from the completed fold-0 run. |
| `run_nnsyn_smoke_training.py`, `run_smoke_inference.py` | Optional three-case installation check. |
| `patches/nnsyn_course_windows.patch` | Guards scheduler-specific signals and fixes two dead imports in the pinned nnsyn checkout. |

Raw scans, model checkpoints, virtual environments, and machine-specific paths are not included.

## Reported baseline

| Item | Value |
|---|---|
| Data | 175 paired SynthRAD2025 Task 1 abdomen cases |
| Split | Fold 0: 140 training and 35 held-out validation cases |
| Plan | `3d_fullres`, patch `48 × 192 × 224`, batch size 2, spacing `3 × 1 × 1 mm` |
| Model | 3D PlainConvUNet with masked MSE |
| Training | 300 epochs, best validation checkpoint |
| Mean / median masked MAE | 105.0 / 102.0 HU |
| Range | 67.4–189.1 HU |
| Device | NVIDIA H100 80 GB GPU |

These are one-fold validation results from the course baseline. They are not the reported SOTA result, a challenge-test score, or an external-site evaluation. The KoalAI algorithm description reports `62.4335 ± 23.2705 HU` for its five-fold ResUNet-MAP ensemble. Reproducing that complete recipe remains separate from the baseline documented here.

## 1. Get the code and data

Clone this course repository, then clone the pinned nnsyn revision beside it:

```bash
git clone https://github.com/zaiyouzy/nnsyn-mini-course.git
cd nnsyn-mini-course
git clone https://github.com/aehrc/nnsyn.git upstream-nnsyn
git -C upstream-nnsyn checkout c3ba6fd8b32f62779f299f78b7d78a96b7fd7695
```

Download SynthRAD2025 Task 1 training data from the [official data page](https://synthrad2025.grand-challenge.org/data/) and accept its terms. After extraction, the abdomen folder should contain:

```text
Task1/AB/
├── 1ABA001/
│   ├── mr.mha
│   ├── ct.mha
│   └── mask.mha
└── ...
```

The course repository does not redistribute these files.

## 2. Create the Python environment

Use Python 3.11 and a CUDA-enabled PyTorch build suitable for the machine. Install the pinned nnsyn checkout and MONAI:

```bash
python -m venv env
source env/bin/activate
python -m pip install --upgrade pip setuptools
python -m pip install -e ./upstream-nnsyn
python -m pip install monai==1.5.2
python -m pip check
```

On Alliance systems, use the available PyTorch and scientific-Python wheels instead of downloading a different CUDA stack. The local and cluster environments used for this project are recorded in [`ENVIRONMENT.md`](ENVIRONMENT.md).

Apply the compatibility patch. It was generated for the pinned commit above:

```bash
git -C upstream-nnsyn apply --check ../code/patches/nnsyn_course_windows.patch
git -C upstream-nnsyn apply ../code/patches/nnsyn_course_windows.patch
```

If CRLF line endings cause whitespace-only conflicts on Windows, use `git apply --ignore-whitespace`. Do not use that option to hide substantive conflicts on another commit.

## 3. Run the full-data baseline

Set two paths:

```bash
export NNSYN_BASE=/path/to/a/large/run/directory
export SYNTHRAD_AB_ROOT=/path/to/extracted/Task1/AB
```

The full runner keeps each stage explicit. This makes failures easier to locate and allows the same commands to be placed inside a cluster job. To run all four stages in one sufficiently long allocation, use:

```bash
python code/full_run/run_full_baseline.py all \
  --base "$NNSYN_BASE" --case-root "$SYNTHRAD_AB_ROOT" --fold 0
```

To run or resume one stage at a time:

```bash
python code/full_run/run_full_baseline.py stage-data \
  --base "$NNSYN_BASE" --case-root "$SYNTHRAD_AB_ROOT"

python code/full_run/run_full_baseline.py preprocess \
  --base "$NNSYN_BASE" --case-root "$SYNTHRAD_AB_ROOT"

python code/full_run/run_full_baseline.py train \
  --base "$NNSYN_BASE" --case-root "$SYNTHRAD_AB_ROOT" --fold 0

python code/full_run/run_full_baseline.py evaluate \
  --base "$NNSYN_BASE" --case-root "$SYNTHRAD_AB_ROOT" --fold 0
```

The training stage installs the included 300-epoch trainer into the active editable nnsyn package and validates with `checkpoint_best.pth`. The evaluation stage expects the validation predictions written by nnsyn and produces:

```text
<NNSYN_BASE>/course_results/300ep_validation/
├── validation_mae.csv
├── validation_summary.json
├── <representative-case>_sCT_HU_masked.mha
└── <representative-case>_representative_result.png
```

### Trillium

The three Slurm files in `full_run/` request one H100 and mirror the completed run. Submit them from this repository after exporting the paths:

```bash
export COURSE_REPO="$PWD"
export NNSYN_BASE=/scratch/$USER/nnsyn_course
export SYNTHRAD_AB_ROOT=/scratch/$USER/nnsyn_course/data/extracted/synthRAD2025_Task1_Train/Task1/AB
export SLURM_ACCOUNT=your-allocation-account

sbatch --account="$SLURM_ACCOUNT" --export=ALL code/full_run/trillium_preprocess.sbatch
# Submit training after preprocessing completes.
sbatch --account="$SLURM_ACCOUNT" --export=ALL code/full_run/trillium_train_300epochs.sbatch
# Submit evaluation after training and validation complete.
sbatch --account="$SLURM_ACCOUNT" --export=ALL code/full_run/trillium_evaluate.sbatch
```

The completed course run used one H100 because this trainer was run as a single-GPU experiment. Requesting more GPUs changes the execution configuration and does not automatically make this exact run comparable.

## 4. Optional local smoke check

The small Windows path uses three cases and a different self-configured plan: patch `56 × 160 × 224`, batch size 2, spacing `3 × 1 × 1 mm`. It is deliberately separated from the full-data plan above.

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
. .\code\setup_course_env.ps1

nnsyn_plan_and_preprocess -d 501 `
  --preprocessing_input MR `
  --preprocessing_target CT `
  --dataset_name Task1_AB_3cases `
  --use_mask

python .\code\run_nnsyn_smoke_training.py --dataset 501 --fold 0 --device cuda
python .\code\run_smoke_inference.py --case 1ABA033 --workers 1
```

This path performs two optimizer updates. Its loss and prediction are diagnostics, not performance results.

## What the baseline establishes

The released pipeline can recreate the data layout, generated plan, 300-epoch masked-MSE trainer, held-out predictions, HU restoration, mask handling, and 35-case MAE table used in the course. It does not yet recreate the complete challenge-winning ResUNet-L, MAP-loss, five-fold ensemble. That distinction is kept explicit in the website, results page, and README.

Please cite [aehrc/nnsyn](https://github.com/aehrc/nnsyn), nnU-Net, and the SynthRAD2025 dataset when using these materials.
