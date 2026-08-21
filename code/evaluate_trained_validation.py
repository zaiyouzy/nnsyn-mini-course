"""Evaluate a trained nnsyn fold and create a masked, representative result figure."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import SimpleITK as sitk


def read_array(path: Path) -> tuple[sitk.Image, np.ndarray]:
    image = sitk.ReadImage(str(path))
    return image, sitk.GetArrayFromImage(image).astype(np.float32)


def make_figure(
    case: str,
    case_dir: Path,
    pred_img: sitk.Image,
    pred_hu: np.ndarray,
    ref_hu: np.ndarray,
    mask: np.ndarray,
    mae: float,
    out_dir: Path,
) -> tuple[Path, Path]:
    _, mri = read_array(case_dir / "mr.mha")
    assert mri.shape == pred_hu.shape == ref_hu.shape == mask.shape

    # This is the repository's intended body-mask post-processing. Values outside
    # the evaluation region become air instead of displaying arbitrary patch output.
    pred_masked = pred_hu.copy()
    pred_masked[~mask] = -1000.0

    restored_path = out_dir / f"{case}_sCT_HU_masked.mha"
    restored_img = sitk.GetImageFromArray(pred_masked)
    restored_img.CopyInformation(pred_img)
    sitk.WriteImage(restored_img, str(restored_path))

    z = int(np.argmax(mask.sum(axis=(1, 2))))
    error = np.abs(pred_hu - ref_hu)
    error[~mask] = np.nan
    mri_values = mri[mask]
    mri_min, mri_max = np.percentile(mri_values, [1, 99])
    error_cmap = plt.get_cmap("magma").copy()
    error_cmap.set_bad("black")

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), facecolor="white")
    axes[0].imshow(mri[z], cmap="gray", vmin=mri_min, vmax=mri_max)
    axes[0].set_title("(a) Input MRI")
    axes[1].imshow(pred_masked[z], cmap="gray", vmin=-500, vmax=1000)
    axes[1].set_title("(b) Predicted synthetic CT")
    axes[2].imshow(ref_hu[z], cmap="gray", vmin=-500, vmax=1000)
    axes[2].set_title("(c) Reference CT")
    im = axes[3].imshow(error[z], cmap=error_cmap, vmin=0, vmax=500)
    axes[3].set_title(f"(d) Absolute error\nMasked MAE = {mae:.1f} HU")
    cbar = fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
    cbar.set_label("Absolute error (HU)")
    for axis in axes:
        axis.axis("off")
    fig.tight_layout()

    figure_path = out_dir / f"{case}_representative_result.png"
    fig.savefig(figure_path, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return restored_path, figure_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        default="Dataset601_SynthRAD2025_Task1_AB_Train",
    )
    parser.add_argument(
        "--trainer",
        default="nnUNetTrainer_nnsyn_loss_masked_300epochs",
    )
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--configuration", default="3d_fullres")
    parser.add_argument("--plans", default="nnUNetPlans")
    parser.add_argument(
        "--case-root",
        type=Path,
        help="Folder containing <case>/mr.mha, ct.mha, and mask.mha.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Defaults to <base>/course_results/300ep_validation.",
    )
    parser.add_argument(
        "--figure-case",
        default="median",
        help="Case identifier, or 'median' for the case nearest cohort median MAE.",
    )
    args = parser.parse_args()

    base = args.base.resolve()
    case_root = args.case_root or (
        base / "data/extracted/synthRAD2025_Task1_Train/Task1/AB"
    )
    output_dir = args.output_dir or (base / "course_results/300ep_validation")
    output_dir.mkdir(parents=True, exist_ok=True)

    result_folder = (
        base
        / "workspace/nnUNet_results"
        / args.dataset
        / f"{args.trainer}__{args.plans}__{args.configuration}"
        / f"fold_{args.fold}"
    )
    validation_dir = result_folder / "validation"
    plan_path = (
        base
        / "workspace/nnUNet_preprocessed"
        / args.dataset
        / "gt_plan/nnUNetPlans.json"
    )

    predictions = sorted(validation_dir.glob("*.mha"))
    if not predictions:
        raise FileNotFoundError(f"No validation predictions found in {validation_dir}")

    plan = json.loads(plan_path.read_text())
    stats = plan["foreground_intensity_properties_per_channel"]["0"]
    ct_mean = float(stats["mean"])
    ct_std = float(stats["std"])

    records: list[dict[str, object]] = []
    arrays: dict[str, tuple[sitk.Image, np.ndarray, np.ndarray, np.ndarray]] = {}

    for pred_path in predictions:
        case = pred_path.stem
        case_dir = case_root / case
        pred_img, pred_norm = read_array(pred_path)
        _, ref_hu = read_array(case_dir / "ct.mha")
        _, mask_values = read_array(case_dir / "mask.mha")
        mask = mask_values > 0
        if pred_norm.shape != ref_hu.shape or ref_hu.shape != mask.shape:
            raise ValueError(
                f"Shape mismatch for {case}: prediction {pred_norm.shape}, "
                f"CT {ref_hu.shape}, mask {mask.shape}"
            )
        pred_hu = pred_norm * ct_std + ct_mean
        mae = float(np.mean(np.abs(pred_hu[mask] - ref_hu[mask])))
        records.append({
            "case": case,
            "masked_mae_hu": mae,
            "body_voxels": int(mask.sum()),
        })
        arrays[case] = (pred_img, pred_hu, ref_hu, mask)

    maes = np.asarray([float(row["masked_mae_hu"]) for row in records])
    summary = {
        "dataset": args.dataset,
        "trainer": args.trainer,
        "fold": args.fold,
        "validation_cases": len(records),
        "ct_mean": ct_mean,
        "ct_std": ct_std,
        "masked_mae_mean_hu": float(maes.mean()),
        "masked_mae_median_hu": float(np.median(maes)),
        "masked_mae_std_hu": float(maes.std(ddof=1)) if len(maes) > 1 else 0.0,
        "masked_mae_min_hu": float(maes.min()),
        "masked_mae_max_hu": float(maes.max()),
    }

    csv_path = output_dir / "validation_mae.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case", "masked_mae_hu", "body_voxels"])
        writer.writeheader()
        writer.writerows(records)

    summary_path = output_dir / "validation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    if args.figure_case == "median":
        median = float(np.median(maes))
        figure_record = min(records, key=lambda row: abs(float(row["masked_mae_hu"]) - median))
        figure_case = str(figure_record["case"])
    else:
        figure_case = args.figure_case
        figure_record = next((row for row in records if row["case"] == figure_case), None)
        if figure_record is None:
            raise ValueError(f"Figure case {figure_case} is not in this validation fold")

    pred_img, pred_hu, ref_hu, mask = arrays[figure_case]
    restored_path, figure_path = make_figure(
        figure_case,
        case_root / figure_case,
        pred_img,
        pred_hu,
        ref_hu,
        mask,
        float(figure_record["masked_mae_hu"]),
        output_dir,
    )

    print("VALIDATION EVALUATION COMPLETED")
    print(f"Cases: {summary['validation_cases']}")
    print(f"Mean masked MAE: {summary['masked_mae_mean_hu']:.1f} HU")
    print(f"Median masked MAE: {summary['masked_mae_median_hu']:.1f} HU")
    print(f"Range: {summary['masked_mae_min_hu']:.1f}–{summary['masked_mae_max_hu']:.1f} HU")
    print(f"Representative case: {figure_case}")
    print(f"CSV: {csv_path}")
    print(f"Summary: {summary_path}")
    print(f"Masked sCT: {restored_path}")
    print(f"Figure: {figure_path}")


if __name__ == "__main__":
    main()
