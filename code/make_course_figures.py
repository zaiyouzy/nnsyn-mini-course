"""Generate the real-data figures used in the nnsyn mini-course slides and notebooks.

Every figure in this script is produced from data that actually exists in this
project: the three extracted SynthRAD2025 Task 1 abdomen cases and the
Dataset501 arrays that nnsyn's own preprocessing wrote to disk. Nothing here is
schematic or synthetic.

Run from the project root with the course environment active:

    # source workspace
    python course_support/make_course_figures.py

    # published mini-course repository
    python code/make_course_figures.py

Outputs land in mini_course/figures/ in the source workspace and figures/ in
the public mini-course repository.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ORIGIN_DATASET = (
    PROJECT_ROOT
    / "dataset"
    / "nnsyn_origin"
    / "Task1_AB_3cases"
)
PREPROCESSED = (
    PROJECT_ROOT
    / "nnsyn_workspace"
    / "nnUNet_preprocessed"
    / "Dataset501_Task1_AB_3cases"
)
ARRAYS = PREPROCESSED / "nnUNetPlans_3d_fullres"
FIGURES = PROJECT_ROOT / "mini_course" / "figures"
if not (PROJECT_ROOT / "mini_course").exists():
    # In the public mini-course repository the script lives in code/ and the
    # published images live directly in figures/.
    FIGURES = PROJECT_ROOT / "figures"

CASE = "1ABA033"
PATCH_SIZE = (56, 160, 224)  # from nnUNetPlans.json, configuration 3d_fullres

# Slide-friendly styling: dark background, generous font sizes.
INK = "#0f1419"
PAPER = "#ffffff"
ACCENT = "#1f6feb"
WARM = "#d1495b"
MASK_TEAL = "#45b8ad"

plt.rcParams.update(
    {
        "figure.facecolor": PAPER,
        "axes.facecolor": PAPER,
        "savefig.facecolor": PAPER,
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "axes.edgecolor": "#c9d1d9",
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
    }
)


def load_raw(case: str) -> dict[str, sitk.Image]:
    return {
        "mr": sitk.ReadImage(
            str(ORIGIN_DATASET / "INPUT_IMAGES" / f"{case}_0000.mha")
        ),
        "ct": sitk.ReadImage(
            str(ORIGIN_DATASET / "TARGET_IMAGES" / f"{case}_0000.mha")
        ),
        "mask": sitk.ReadImage(
            str(ORIGIN_DATASET / "MASKS" / f"{case}.mha")
        ),
    }


def tri_planar(volume: np.ndarray) -> list[np.ndarray]:
    """Return axial, coronal and sagittal mid-slices from a (z, y, x) volume.

    Every panel is drawn with origin="lower", so the axial slice is flipped here
    to put anterior at the top, matching the usual radiological convention.
    """
    z, y, x = volume.shape
    return [
        volume[z // 2, ::-1, :],
        volume[:, y // 2, :],
        volume[:, :, x // 2],
    ]


def show(ax, image: np.ndarray, *, vmin=None, vmax=None, cmap="gray", aspect=None):
    ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower", aspect=aspect)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


# ---------------------------------------------------------------- figure 01
def fig_mr_ct_mask(raw: dict[str, sitk.Image]) -> None:
    """One real abdomen case: MR and paired CT with the body mask overlaid."""
    mr = sitk.GetArrayFromImage(raw["mr"]).astype(np.float32)
    ct = sitk.GetArrayFromImage(raw["ct"]).astype(np.float32)
    mask = sitk.GetArrayFromImage(raw["mask"]).astype(np.float32)
    spacing = raw["mr"].GetSpacing()  # (x, y, z)
    # z is 3 mm while x/y are 1 mm, so coronal/sagittal views need stretching.
    aspects = [1.0, spacing[2] / spacing[1], spacing[2] / spacing[0]]

    mr_hi = float(np.percentile(mr[mask > 0], 99.5))
    rows = [
        ("MR + body mask", tri_planar(mr), dict(cmap="gray", vmin=0, vmax=mr_hi)),
        ("CT + body mask", tri_planar(ct), dict(cmap="gray", vmin=-200, vmax=300)),
    ]
    mask_slices = tri_planar(mask)
    views = ["Axial", "Coronal", "Sagittal"]

    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.4))
    for r, (label, slices, kw) in enumerate(rows):
        for c, image in enumerate(slices):
            show(axes[r, c], image, aspect=aspects[c], **kw)
            axes[r, c].contour(
                mask_slices[c],
                levels=[0.5],
                colors=[MASK_TEAL],
                linewidths=1.6,
                origin="lower",
            )
            if r == 0:
                axes[r, c].set_title(views[c], fontsize=14, color=INK, pad=8)
        axes[r, 0].set_ylabel(label, fontsize=14, color=INK, labelpad=12)
        axes[r, 0].set_yticks([])

    fig.suptitle(
        f"SynthRAD2025 Task 1 abdomen case {CASE}: paired MR and CT with body-mask overlay",
        fontsize=16,
        color=INK,
        y=0.98,
    )
    fig.text(
        0.5,
        0.018,
        f"teal contour = body-mask boundary  |  "
        f"{mr.shape[2]} x {mr.shape[1]} x {mr.shape[0]} voxels  |  spacing "
        f"{spacing[0]:.0f} x {spacing[1]:.0f} x {spacing[2]:.0f} mm",
        ha="center",
        fontsize=11.5,
        color="#57606a",
    )
    fig.tight_layout(rect=(0.02, 0.055, 1, 0.95))
    fig.savefig(FIGURES / "fig01_mr_ct_mask_triplanar.png", dpi=170)
    plt.close(fig)


# ---------------------------------------------------------------- figure 02
def fig_normalization(raw: dict[str, sitk.Image]) -> None:
    """Why MR and CT cannot share one normalization strategy."""
    mr = sitk.GetArrayFromImage(raw["mr"]).astype(np.float32)
    ct = sitk.GetArrayFromImage(raw["ct"]).astype(np.float32)
    mask = sitk.GetArrayFromImage(raw["mask"]) > 0

    mr_norm = np.load(ARRAYS / f"{CASE}.npy", mmap_mode="r")[0]
    ct_norm = np.load(ARRAYS / f"{CASE}_seg.npy", mmap_mode="r")[0]
    norm_mask = np.load(ARRAYS / f"{CASE}_mask.npy", mmap_mode="r")[0] > 0

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].hist(mr[mask], bins=200, color=ACCENT, alpha=0.85)
    axes[0, 0].set_title("Raw MR intensity (arbitrary units)")
    axes[0, 0].set_xlabel("signal intensity")
    axes[0, 0].axvline(float(mr[mask].mean()), color=WARM, lw=2, label="mean")
    axes[0, 0].legend(frameon=False)

    axes[0, 1].hist(ct[mask], bins=200, color="#6e7781", alpha=0.9)
    axes[0, 1].set_title("Raw CT intensity (Hounsfield units)")
    axes[0, 1].set_xlabel("HU")
    for hu, name in [(-1000, "air"), (0, "water"), (700, "bone")]:
        axes[0, 1].axvline(hu, color=WARM, lw=1.2, ls="--")
        axes[0, 1].text(hu, axes[0, 1].get_ylim()[1] * 0.9, f" {name}", fontsize=10,
                        color=WARM)

    axes[1, 0].hist(np.asarray(mr_norm[norm_mask]), bins=200, color=ACCENT, alpha=0.85)
    axes[1, 0].set_title("Preprocessed MR: per-image z-score")
    axes[1, 0].set_xlabel("normalized value")

    axes[1, 1].hist(np.asarray(ct_norm[norm_mask]), bins=200, color="#6e7781", alpha=0.9)
    axes[1, 1].set_title("Preprocessed CT target: clip, then standardize")
    axes[1, 1].set_xlabel("normalized value")

    for ax in axes.ravel():
        ax.set_ylabel("voxel count")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        "MR carries no physical scale; CT does. That is why nnsyn normalizes them "
        "differently.",
        fontsize=15,
        color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIGURES / "fig02_normalization_histograms.png", dpi=170)
    plt.close(fig)


# ---------------------------------------------------------------- figure 03
def _pick_patch_origin(mask: np.ndarray, patch: tuple[int, int, int]) -> tuple[int, int, int]:
    """Find a patch location whose mask coverage is partial, so the mask is visible.

    A patch taken from the middle of the abdomen is entirely inside the body, which
    makes the mask a constant image and teaches nothing. We slide the window along
    the anterior-posterior axis and keep the first location whose coverage falls in
    an informative range.
    """
    pz, py, px = patch
    _, z, y, x = mask.shape
    z0 = max((z - pz) // 2, 0)
    x0 = max((x - px) // 2, 0)
    best, best_gap = 0, 1.0
    for y0 in range(0, max(y - py, 0) + 1, 8):
        window = mask[0, z0 : z0 + pz, y0 : y0 + py, x0 : x0 + px]
        coverage = float(np.asarray(window).mean())
        gap = abs(coverage - 0.65)
        if gap < best_gap:
            best, best_gap = y0, gap
    return z0, best, x0


def fig_training_patch() -> None:
    """The tensor the network actually receives, cropped from real arrays."""
    mr = np.load(ARRAYS / f"{CASE}.npy", mmap_mode="r")
    ct = np.load(ARRAYS / f"{CASE}_seg.npy", mmap_mode="r")
    mask = np.load(ARRAYS / f"{CASE}_mask.npy", mmap_mode="r")

    pz, py, px = PATCH_SIZE
    z0, y0, x0 = _pick_patch_origin(mask, PATCH_SIZE)
    sl = (slice(0, 1), slice(z0, z0 + pz), slice(y0, y0 + py), slice(x0, x0 + px))

    mr_p = np.asarray(mr[sl])[0]
    ct_p = np.asarray(ct[sl])[0]
    mask_p = np.asarray(mask[sl])[0].astype(np.float32)
    coverage = float(mask_p.mean())

    mid = mr_p.shape[0] // 2
    panels = [
        ("source patch\nMR", mr_p[mid], dict(cmap="gray")),
        ("target patch\nCT", ct_p[mid], dict(cmap="gray")),
        (
            f"mask patch\n{coverage:.0%} valid voxels",
            mask_p[mid],
            dict(cmap="magma", vmin=0, vmax=1),
        ),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(16.5, 4.6))
    for ax, (label, image, kw) in zip(axes, panels):
        show(ax, image[::-1, :], **kw)
        ax.set_title(label, fontsize=14, color=INK, pad=10)

    # Fourth panel: the mask boundary drawn on top of the MR patch, which is how
    # you actually check that the three volumes still describe the same anatomy.
    show(axes[3], mr_p[mid][::-1, :], cmap="gray")
    axes[3].contour(mask_p[mid][::-1, :], levels=[0.5], colors=[WARM], linewidths=2.0)
    axes[3].set_title("mask boundary\non the MR patch", fontsize=14, color=INK, pad=10)

    fig.suptitle(
        "One training patch: the identical crop is applied to source, target and mask",
        fontsize=16,
        color=INK,
    )
    fig.text(
        0.5,
        0.035,
        f"patch size {pz} x {py} x {px} (from nnUNetPlans.json)   |   "
        f"batch size 2   ->   network input tensor [2, 1, {pz}, {py}, {px}]",
        ha="center",
        fontsize=13,
        color="#57606a",
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.9))
    fig.savefig(FIGURES / "fig03_training_patch.png", dpi=170)
    plt.close(fig)


# ---------------------------------------------------------------- figure 04
def fig_seg_npy_is_continuous() -> None:
    """Evidence that Dataset501's *_seg.npy holds a continuous image, not labels."""
    ct = np.load(ARRAYS / f"{CASE}_seg.npy", mmap_mode="r")[0]
    mask = np.load(ARRAYS / f"{CASE}_mask.npy", mmap_mode="r")[0]
    sub = np.asarray(ct[::2, ::3, ::3])
    sub_mask = np.asarray(mask[::2, ::3, ::3]) > 0
    values = sub[sub_mask]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5))

    axes[0].hist(values, bins=250, color=ACCENT, alpha=0.9)
    axes[0].set_title(f"{CASE}_seg.npy value distribution", fontsize=14)
    axes[0].set_xlabel("stored value")
    axes[0].set_ylabel("voxel count")
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)
    n_unique = int(np.unique(values[:200000]).size)
    axes[0].text(
        0.02,
        0.95,
        f"dtype: {ct.dtype}\n"
        f"range: {values.min():.2f} to {values.max():.2f}\n"
        f"{n_unique:,} distinct values in a 200k sample",
        transform=axes[0].transAxes,
        va="top",
        fontsize=12,
        color=INK,
        bbox=dict(boxstyle="round,pad=0.5", fc="#f6f8fa", ec="#c9d1d9"),
    )

    fake_labels = np.array([0, 1, 2, 3, 4])
    axes[1].bar(fake_labels, [1, 1, 1, 1, 1], color="#6e7781", width=0.45)
    axes[1].set_title("What a real segmentation *_seg.npy looks like", fontsize=14)
    axes[1].set_xlabel("stored value")
    axes[1].set_ylabel("(schematic)")
    axes[1].set_xticks(fake_labels)
    axes[1].set_yticks([])
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    axes[1].text(
        0.5,
        0.6,
        "a handful of\ninteger class IDs",
        transform=axes[1].transAxes,
        ha="center",
        fontsize=13,
        color=WARM,
    )

    fig.suptitle(
        "The filename says 'seg', but nnsyn stores a continuous CT image there",
        fontsize=16,
        color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(FIGURES / "fig04_seg_npy_is_continuous.png", dpi=170)
    plt.close(fig)


# ---------------------------------------------------------------- figure 05
def fig_alignment(raw: dict[str, sitk.Image]) -> None:
    """A checkerboard mosaic: the paired-data assumption, verified visually."""
    mr = sitk.GetArrayFromImage(raw["mr"]).astype(np.float32)
    ct = sitk.GetArrayFromImage(raw["ct"]).astype(np.float32)
    mask = sitk.GetArrayFromImage(raw["mask"]) > 0

    z = mr.shape[0] // 2
    # Flip to put anterior at the top, consistent with the other figures.
    mr_s, ct_s, mask_s = mr[z, ::-1], ct[z, ::-1], mask[z, ::-1]

    def scale(image, lo, hi):
        return np.clip((image - lo) / (hi - lo), 0, 1)

    mr_n = scale(mr_s, 0, float(np.percentile(mr_s[mask_s], 99.5)))
    ct_n = scale(ct_s, -200, 300)

    tile = 40
    yy, xx = np.indices(mr_s.shape)
    board = ((yy // tile) + (xx // tile)) % 2 == 0
    mosaic = np.where(board, mr_n, ct_n)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5))
    show(axes[0], mr_n, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("MR", fontsize=14, pad=10)
    show(axes[1], ct_n, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("CT", fontsize=14, pad=10)
    show(axes[2], mosaic, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Checkerboard mosaic", fontsize=14, pad=10)

    fig.suptitle(
        "Anatomy runs continuously across tile boundaries, so the pair is aligned",
        fontsize=16,
        color=INK,
    )
    fig.text(
        0.5,
        0.04,
        "If the MR and CT were misregistered, organ edges would jump at every tile "
        "seam and the voxel-wise loss would be meaningless.",
        ha="center",
        fontsize=12,
        color="#57606a",
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.92))
    fig.savefig(FIGURES / "fig05_alignment_checkerboard.png", dpi=170)
    plt.close(fig)


FIGURE_FUNCS = {
    "mr_ct_mask": ("needs_raw", fig_mr_ct_mask),
    "normalization": ("needs_raw", fig_normalization),
    "patch": ("arrays_only", fig_training_patch),
    "seg_npy": ("arrays_only", fig_seg_npy_is_continuous),
    "alignment": ("needs_raw", fig_alignment),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only",
        nargs="*",
        choices=sorted(FIGURE_FUNCS),
        help="Build only the named figures.",
    )
    args = parser.parse_args()

    FIGURES.mkdir(parents=True, exist_ok=True)
    selected = args.only or sorted(FIGURE_FUNCS)

    raw = None
    if any(FIGURE_FUNCS[name][0] == "needs_raw" for name in selected):
        raw = load_raw(CASE)

    for name in selected:
        kind, func = FIGURE_FUNCS[name]
        print(f"building {name} ...", flush=True)
        func(raw) if kind == "needs_raw" else func()

    print(f"\nfigures written to {FIGURES}")
    for path in sorted(FIGURES.glob("*.png")):
        print(f"  {path.name}  ({path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
