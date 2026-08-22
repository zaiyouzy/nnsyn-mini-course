"""Run the full-data baseline reported in the nnsyn mini-course.

Stages are explicit so the same entry point works interactively or inside a
cluster job. Run ``python run_full_baseline.py --help`` for the required paths.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


DATASET_ID = 601
DATASET_NAME = "SynthRAD2025_Task1_AB_Train"
TRAINER = "nnUNetTrainer_nnsyn_loss_masked_300epochs"
CONFIGURATION = "3d_fullres"
PLANS = "nnUNetPlans"


def command_exists(name: str) -> bool:
    return shutil.which(name) is not None


def run(command: list[str], env: dict[str, str]) -> None:
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, env=env, check=True)


def build_environment(base: Path, script_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["nnUNet_raw"] = str(base / "workspace/nnUNet_raw")
    env["nnUNet_preprocessed"] = str(base / "workspace/nnUNet_preprocessed")
    env["nnUNet_results"] = str(base / "workspace/nnUNet_results")
    env["nnsyn_origin_dataset"] = str(
        base / "data/nnsyn_origin" / DATASET_NAME
    )
    env["MPLCONFIGDIR"] = str(base / "cache/matplotlib")
    env["nnUNet_n_proc_DA"] = env.get("nnUNet_n_proc_DA", "0")
    env["nnUNet_def_n_proc"] = env.get("nnUNet_def_n_proc", "2")
    env["PYTHONUNBUFFERED"] = "1"
    compat = str(script_dir / "compat")
    env["PYTHONPATH"] = compat + os.pathsep + env.get("PYTHONPATH", "")
    for key in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results", "MPLCONFIGDIR"):
        Path(env[key]).mkdir(parents=True, exist_ok=True)
    return env


def install_trainer(script_dir: Path) -> Path:
    import nnunetv2

    trainer_dir = (
        Path(nnunetv2.__path__[0])
        / "training/nnUNetTrainer/variants/nnsyn"
    )
    destination = trainer_dir / f"{TRAINER}.py"
    shutil.copy2(script_dir / f"{TRAINER}.py", destination)
    print(f"Trainer installed at {destination}")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=("stage-data", "preprocess", "train", "evaluate", "all"),
    )
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument(
        "--case-root",
        type=Path,
        required=True,
        help="Extracted SynthRAD abdomen folder containing <case>/{mr,ct,mask}.mha.",
    )
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument(
        "--transfer-mode",
        choices=("copy", "hardlink", "symlink"),
        default="copy",
    )
    args = parser.parse_args()

    base = args.base.resolve()
    case_root = args.case_root.resolve()
    script_dir = Path(__file__).resolve().parent
    code_dir = script_dir.parent
    env = build_environment(base, script_dir)

    stages = (
        ("stage-data", "preprocess", "train", "evaluate")
        if args.stage == "all"
        else (args.stage,)
    )

    for stage in stages:
        if stage == "stage-data":
            destination = base / "data/nnsyn_origin" / DATASET_NAME
            run(
                [
                    sys.executable,
                    str(script_dir / "prepare_synthrad_ab.py"),
                    "--source",
                    str(case_root),
                    "--destination",
                    str(destination),
                    "--mode",
                    args.transfer_mode,
                ],
                env,
            )

        elif stage == "preprocess":
            if not command_exists("nnsyn_plan_and_preprocess"):
                raise RuntimeError("nnsyn_plan_and_preprocess is not on PATH")
            run(
                [
                    "nnsyn_plan_and_preprocess",
                    "-d",
                    str(DATASET_ID),
                    "-c",
                    CONFIGURATION,
                    "-pl",
                    "ExperimentPlanner",
                    "-p",
                    PLANS,
                    "--preprocessing_input",
                    "MR",
                    "--preprocessing_target",
                    "CT",
                    "--dataset_name",
                    DATASET_NAME,
                    "--use_mask",
                ],
                env,
            )

        elif stage == "train":
            if not command_exists("nnsyn_train"):
                raise RuntimeError("nnsyn_train is not on PATH")
            try:
                import monai  # noqa: F401
            except ImportError as exc:
                raise RuntimeError(
                    "MONAI is required because the pinned nnsyn checkout imports "
                    "its SSIM module during trainer discovery. Install monai==1.5.2."
                ) from exc
            install_trainer(script_dir)
            run(
                [
                    "nnsyn_train",
                    str(DATASET_ID),
                    CONFIGURATION,
                    str(args.fold),
                    "-tr",
                    TRAINER,
                    "-p",
                    PLANS,
                    "-num_gpus",
                    "1",
                    "-device",
                    "cuda",
                    "--val_best",
                ],
                env,
            )

        elif stage == "evaluate":
            run(
                [
                    sys.executable,
                    str(code_dir / "evaluate_trained_validation.py"),
                    "--base",
                    str(base),
                    "--case-root",
                    str(case_root),
                    "--fold",
                    str(args.fold),
                ],
                env,
            )


if __name__ == "__main__":
    main()
