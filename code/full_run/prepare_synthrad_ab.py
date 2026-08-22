"""Stage SynthRAD2025 Task 1 abdomen cases for nnsyn.

The official archive stores each case as <case>/{mr,ct,mask}.mha. nnsyn expects
three flat folders with a shared case identifier. This script performs that
deterministic conversion without changing the image contents.
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


def transfer(source: Path, destination: Path, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    if mode == "copy":
        shutil.copy2(source, destination)
    elif mode == "hardlink":
        os.link(source, destination)
    else:
        destination.symlink_to(source.resolve())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Folder containing <case>/mr.mha, ct.mha, and mask.mha.",
    )
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("copy", "hardlink", "symlink"),
        default="copy",
        help="How to stage the files. Copy is the most portable default.",
    )
    args = parser.parse_args()

    source = args.source.resolve()
    destination = args.destination.resolve()
    case_dirs = sorted(path for path in source.iterdir() if path.is_dir())
    valid_cases: list[tuple[str, Path, Path, Path]] = []

    for case_dir in case_dirs:
        files = (case_dir / "mr.mha", case_dir / "ct.mha", case_dir / "mask.mha")
        if all(path.is_file() for path in files):
            valid_cases.append((case_dir.name, *files))

    if not valid_cases:
        raise FileNotFoundError(
            f"No complete cases were found under {source}. Expected "
            "<case>/mr.mha, <case>/ct.mha, and <case>/mask.mha."
        )

    for folder in ("INPUT_IMAGES", "TARGET_IMAGES", "MASKS"):
        (destination / folder).mkdir(parents=True, exist_ok=True)

    for case, mr_path, ct_path, mask_path in valid_cases:
        transfer(mr_path, destination / "INPUT_IMAGES" / f"{case}_0000.mha", args.mode)
        transfer(ct_path, destination / "TARGET_IMAGES" / f"{case}_0000.mha", args.mode)
        transfer(mask_path, destination / "MASKS" / f"{case}.mha", args.mode)

    counts = {
        folder: len(list((destination / folder).glob("*.mha")))
        for folder in ("INPUT_IMAGES", "TARGET_IMAGES", "MASKS")
    }
    expected = len(valid_cases)
    if any(count != expected for count in counts.values()):
        raise RuntimeError(f"Staging count mismatch: expected {expected}, found {counts}")

    print(f"STAGING COMPLETED: {expected} cases")
    print(f"Destination: {destination}")
    print(f"Transfer mode: {args.mode}")


if __name__ == "__main__":
    main()
