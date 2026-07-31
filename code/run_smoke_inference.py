"""Generate a synthetic CT from the smoke checkpoint and build the four-panel figure.

IMPORTANT, and this is a teaching point in itself:
we deliberately do NOT call the repo's own ``nnsyn_predict`` entry point here.
``nnsyn_predict_entrypoints.py`` declares a large argparse interface
(``-step_size``, ``--disable_tta``, ``-device``, ``-npp``, ``-nps``, ``--rec``,
``--verbose`` and more) but the line that actually runs prediction is::

    os.system(f"nnUNetv2_predict -d {args.d} -i {args.i} -o {args.o} "
              f"-c {args.c} -p {args.p} -tr {args.tr} -f {args.f} -chk {args.chk} ")

Every other flag is parsed and then silently dropped. On Windows we need to
lower the number of export/preprocessing worker processes, so we call
``nnUNetv2_predict`` directly and then reuse the repository's own
``revert_normalisation`` for post-processing. That keeps the science identical
while making the run survivable on a laptop.

The checkpoint used here saw two optimizer steps. The output is a pipeline
demonstration, not a trained model.

Usage (from the project root, with the course environment active)::

    # source workspace
    python course_support/run_smoke_inference.py

    # published mini-course repository
    python code/run_smoke_inference.py --case 1ABA033 --workers 1
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_ID = "501"
DATASET_NAME = "Dataset501_Task1_AB_3cases"
CONFIGURATION = "3d_fullres"
PLANS = "nnUNetPlans"
TRAINER = "nnUNetTrainer_nnsyn_smoke"
FOLD = "0"
CHECKPOINT = "checkpoint_final.pth"

# Fold 0 holds out 1ABA033 (see splits_final.json), so this case was never
# used to update the weights.
DEFAULT_CASE = "1ABA033"

WORKSPACE = PROJECT_ROOT / "nnsyn_workspace"
ORIGIN_DATASET = (
    PROJECT_ROOT
    / "dataset"
    / "nnsyn_origin"
    / "Task1_AB_3cases"
)
PREDICT_ROOT = WORKSPACE / "course_predictions"
FIGURES = PROJECT_ROOT / "mini_course" / "figures"
if not (PROJECT_ROOT / "mini_course").exists():
    # In the public mini-course repository the script lives in code/ and the
    # published images live directly in figures/.
    FIGURES = PROJECT_ROOT / "figures"


def case_paths(case: str) -> tuple[Path, Path, Path]:
    """Return the staged MR, CT, and mask paths for one origin-dataset case."""
    return (
        ORIGIN_DATASET / "INPUT_IMAGES" / f"{case}_0000.mha",
        ORIGIN_DATASET / "TARGET_IMAGES" / f"{case}_0000.mha",
        ORIGIN_DATASET / "MASKS" / f"{case}.mha",
    )


def build_input_folders(case: str) -> tuple[Path, Path]:
    """Stage the single held-out case into the folder layout inference expects."""
    in_dir = PREDICT_ROOT / f"{case}_input"
    mask_dir = PREDICT_ROOT / f"{case}_mask"
    in_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    # The channel suffix _0000 is required: inference reuses the training reader.
    mr_path, _, mask_path = case_paths(case)
    shutil.copy2(mr_path, in_dir / f"{case}_0000.mha")
    shutil.copy2(mask_path, mask_dir / f"{case}.mha")
    return in_dir, mask_dir


def run_prediction(in_dir: Path, out_dir: Path, workers: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "nnUNetv2_predict",
        "-d", DATASET_ID,
        "-i", str(in_dir),
        "-o", str(out_dir),
        "-c", CONFIGURATION,
        "-p", PLANS,
        "-tr", TRAINER,
        "-f", FOLD,
        "-chk", CHECKPOINT,
        # These are exactly the flags nnsyn_predict would have thrown away.
        "-npp", str(workers),
        "-nps", str(workers),
        "--disable_tta",
    ]
    print("running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def revert(out_dir: Path, mask_dir: Path) -> Path:
    """Undo the CT z-score using the repository's own helper."""
    from nnunetv2.analysis.revert_normalisation import (
        get_ct_normalisation_values,
        revert_normalisation,
    )

    ct_plan = WORKSPACE / "nnUNet_preprocessed" / DATASET_NAME / "gt_plan" / f"{PLANS}.json"
    ct_mean, ct_std = get_ct_normalisation_values(str(ct_plan))

    reverted = Path(str(out_dir) + "_revert_norm")
    revert_normalisation(
        str(out_dir),
        ct_mean,
        ct_std,
        save_path=str(reverted),
        mask_path=str(mask_dir),
        mask_outside_value=-1000,
    )
    return reverted


def make_figure(case: str, sct_path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import SimpleITK as sitk

    def read(path: Path) -> np.ndarray:
        return sitk.GetArrayFromImage(sitk.ReadImage(str(path))).astype(np.float32)

    mr_path, ct_path, mask_path = case_paths(case)
    mr = read(mr_path)
    ct = read(ct_path)
    mask = read(mask_path) > 0
    sct = read(sct_path)

    if sct.shape != ct.shape:
        raise SystemExit(
            f"shape mismatch: prediction {sct.shape} vs ground-truth CT {ct.shape}"
        )

    z = mr.shape[0] // 2
    flip = lambda a: a[z, ::-1, :]  # noqa: E731  anterior at the top
    mr_s, ct_s, sct_s, mask_s = flip(mr), flip(ct), flip(sct), flip(mask)

    error = np.abs(sct_s - ct_s)
    error[~mask_s] = 0.0

    mae = float(np.abs(sct[mask] - ct[mask]).mean())
    print(f"\nmasked MAE = {mae:.1f} HU  (smoke checkpoint, NOT a trained model)")

    fig, axes = plt.subplots(1, 4, figsize=(17.5, 5.2))
    panels = [
        ("Input MR", mr_s, dict(cmap="gray", vmin=0,
                                vmax=float(np.percentile(mr_s[mask_s], 99.5)))),
        ("Smoke synthetic CT", sct_s, dict(cmap="gray", vmin=-200, vmax=300)),
        ("Ground-truth CT", ct_s, dict(cmap="gray", vmin=-200, vmax=300)),
    ]
    for ax, (title, image, kw) in zip(axes, panels):
        ax.imshow(image, origin="lower", **kw)
        ax.set_title(title, fontsize=15, pad=10)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    im = axes[3].imshow(error, origin="lower", cmap="inferno", vmin=0, vmax=1000)
    axes[3].set_title("Absolute error (HU)", fontsize=15, pad=10)
    axes[3].set_xticks([])
    axes[3].set_yticks([])
    for spine in axes[3].spines.values():
        spine.set_visible(False)
    fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)

    fig.suptitle(
        f"Case {case} (held out in fold 0)   |   masked MAE {mae:.0f} HU",
        fontsize=16,
    )
    fig.text(
        0.5,
        0.035,
        "Two-iteration smoke checkpoint - pipeline demonstration only, "
        "NOT a trained clinical model. These numbers must not be read as performance.",
        ha="center",
        fontsize=12.5,
        color="#b3261e",
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.93))

    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / "fig06_smoke_synthetic_ct.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default=DEFAULT_CASE)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Worker processes for preprocessing and export. Keep at 1 on Windows.",
    )
    parser.add_argument(
        "--figure-only",
        action="store_true",
        help="Skip inference and rebuild the figure from an existing prediction.",
    )
    args = parser.parse_args()

    for var in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"):
        if var not in os.environ:
            sys.exit(f"{var} is not set. Run setup_nnsyn_course_env.ps1 first.")

    out_dir = PREDICT_ROOT / f"{args.case}_smoke_pred"
    reverted = Path(str(out_dir) + "_revert_norm")

    if not args.figure_only:
        in_dir, mask_dir = build_input_folders(args.case)
        run_prediction(in_dir, out_dir, args.workers)
        reverted = revert(out_dir, mask_dir)

    sct_path = reverted / f"{args.case}.mha"
    if not sct_path.exists():
        candidates = sorted(reverted.glob("*.mha"))
        if not candidates:
            sys.exit(f"no prediction found in {reverted}")
        sct_path = candidates[0]

    figure = make_figure(args.case, sct_path)
    print(f"\nsynthetic CT : {sct_path}")
    print(f"figure       : {figure}")


if __name__ == "__main__":
    main()
