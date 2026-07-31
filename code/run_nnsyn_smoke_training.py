"""Run the bounded nnsyn course trainer without recursive trainer discovery."""

import argparse
import os
from pathlib import Path

import torch

from batchgenerators.utilities.file_and_folder_operations import load_json, save_json
from nnunetv2.paths import nnUNet_preprocessed
from nnunetv2.training.nnUNetTrainer.variants.nnsyn.nnUNetTrainer_nnsyn_smoke import (
    nnUNetTrainer_nnsyn_smoke,
)
from nnunetv2.utilities.dataset_name_id_conversion import (
    maybe_convert_to_dataset_name,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="501")
    parser.add_argument("--configuration", default="3d_fullres")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--plans", default="nnUNetPlans")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--num-da-workers",
        type=int,
        default=0,
        help=(
            "Data-augmentation worker processes. The course default is 0 "
            "(single-process), which is reliable on Windows."
        ),
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate configuration and trainer construction without training.",
    )
    return parser.parse_args()


def ensure_course_split(
    dataset_folder: Path,
    plans: dict,
    configuration: str,
    fold: int,
) -> None:
    """Create deterministic leave-one-out splits for tiny course datasets."""
    split_file = dataset_folder / "splits_final.json"
    if split_file.exists():
        return

    data_identifier = plans["configurations"][configuration]["data_identifier"]
    case_folder = dataset_folder / data_identifier
    case_ids = sorted(path.stem for path in case_folder.glob("*.npz"))

    if len(case_ids) == 1:
        raise RuntimeError(
            "The smoke dataset has only one case, so a separate validation "
            "case cannot be created."
        )
    if 1 < len(case_ids) < 5:
        splits = [
            {
                "train": [case_id for case_id in case_ids if case_id != val_case],
                "val": [val_case],
            }
            for val_case in case_ids
        ]
        save_json(splits, split_file, sort_keys=False)
        print(
            f"Created {len(splits)} deterministic leave-one-out course splits "
            f"at {split_file}"
        )
        if fold >= len(splits):
            raise ValueError(
                f"Fold {fold} was requested, but this {len(case_ids)}-case "
                f"course dataset provides folds 0-{len(splits) - 1}."
            )


def main() -> None:
    args = parse_args()
    os.environ["nnUNet_n_proc_DA"] = str(args.num_da_workers)
    dataset_name = maybe_convert_to_dataset_name(args.dataset)
    dataset_folder = Path(nnUNet_preprocessed) / dataset_name
    plans = load_json(dataset_folder / f"{args.plans}.json")
    dataset_json = load_json(dataset_folder / "dataset.json")
    ensure_course_split(dataset_folder, plans, args.configuration, args.fold)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but PyTorch cannot see the NVIDIA GPU."
        )

    trainer = nnUNetTrainer_nnsyn_smoke(
        plans=plans,
        configuration=args.configuration,
        fold=args.fold,
        dataset_json=dataset_json,
        unpack_dataset=True,
        device=torch.device(args.device),
    )

    print(f"Dataset: {dataset_name}")
    print(f"Configuration: {args.configuration}")
    print(f"Device: {trainer.device}")
    print(f"Epochs: {trainer.num_epochs}")
    print(f"Training iterations: {trainer.num_iterations_per_epoch}")
    print(f"Validation iterations: {trainer.num_val_iterations_per_epoch}")
    print(f"Data-augmentation workers: {args.num_da_workers}")

    if not args.check_only:
        trainer.run_training()


if __name__ == "__main__":
    main()
