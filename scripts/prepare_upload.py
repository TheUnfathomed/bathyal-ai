"""Summarize the merged training dataset and create a tar archive for upload."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def SummarizeDataset(dataset_dir: Path) -> dict:
    TrainDir = dataset_dir / "train"
    ValDir = dataset_dir / "val"
    Summary: dict[str, dict] = {}

    for split_name, split_dir in [("train", TrainDir), ("val", ValDir)]:
        if not split_dir.exists():
            continue
        for species_dir in sorted(split_dir.iterdir()):
            if not species_dir.is_dir():
                continue
            Count = len([f for f in species_dir.iterdir() if f.is_file()])
            if species_dir.name not in Summary:
                Summary[species_dir.name] = {"train": 0, "val": 0, "total": 0}
            Summary[species_dir.name][split_name] = Count
            Summary[species_dir.name]["total"] += Count

    return Summary


def main() -> None:
    import argparse

    Parser = argparse.ArgumentParser(description="Summarize and tar training dataset for upload.")
    Parser.add_argument("--dataset-dir", type=Path, default=Path("D:/bathyal_training_data"))
    Parser.add_argument("--output-tar", type=Path, default=Path("D:/bathyal_training_data.tar.gz"))
    Parser.add_argument("--min-images", type=int, default=5, help="Only report species with at least this many images.")
    Parser.add_argument("--tar", action="store_true", help="Create tar archive for upload.")
    Args = Parser.parse_args()

    print("Scanning dataset...")
    Summary = SummarizeDataset(Args.dataset_dir)

    TotalImages = sum(s["total"] for s in Summary.values())
    TotalSpecies = len(Summary)
    UsableSpecies = sum(1 for s in Summary.values() if s["total"] >= Args.min_images)

    print(f"\n{'='*60}")
    print(f"Dataset Summary")
    print(f"{'='*60}")
    print(f"Total images:        {TotalImages:,}")
    print(f"Total species:       {TotalSpecies}")
    print(f"Usable species (>={Args.min_images} imgs): {UsableSpecies}")
    print(f"{'='*60}")

    Sorted = sorted(Summary.items(), key=lambda x: x[1]["total"], reverse=True)

    print(f"\nTop 30 species:")
    for i, (name, counts) in enumerate(Sorted[:30], 1):
        print(f"  {i:3d}. {name:<45s} train={counts['train']:>5d}  val={counts['val']:>4d}  total={counts['total']:>5d}")

    print(f"\nBottom 10 species (>={Args.min_images}):")
    Usable = [(n, c) for n, c in Sorted if c["total"] >= Args.min_images]
    for name, counts in Usable[-10:]:
        print(f"       {name:<45s} train={counts['train']:>5d}  val={counts['val']:>4d}  total={counts['total']:>5d}")

    ThinSpecies = [(n, c) for n, c in Sorted if c["total"] < Args.min_images]
    if ThinSpecies:
        print(f"\n  {len(ThinSpecies)} species with <{Args.min_images} images (will be filtered at training time)")

    if Args.tar:
        print(f"\nCreating tar archive at {Args.output_tar}...")
        subprocess.run(
            ["tar", "czf", str(Args.output_tar), "-C", str(Args.dataset_dir.parent), Args.dataset_dir.name],
            check=True,
        )
        TarSize = Args.output_tar.stat().st_size / (1024 ** 3)
        print(f"Done: {Args.output_tar} ({TarSize:.1f} GB)")
        print(f"\nUpload to Google Drive, then on vast.ai:")
        print(f"  # Install gdown or rclone, then:")
        print(f"  # gdown <gdrive_file_id> -O bathyal_training_data.tar.gz")
        print(f"  # tar xzf bathyal_training_data.tar.gz")


if __name__ == "__main__":
    main()
