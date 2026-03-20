"""Download SEAMAPD21 dataset and extract fish crops into training folder structure."""

from __future__ import annotations

import csv
import hashlib
import re
import subprocess
import sys
from pathlib import Path

from PIL import Image

SEAMAPD21_PARTS = [
    f"https://grunt.sefsc.noaa.gov/parr/SEAMAPD21.tar.gz.{suffix}"
    for suffix in [
        "aa", "ab", "ac", "ad", "ae", "af", "ag", "ah", "ai", "aj",
        "ak", "al", "am", "an", "ao", "ap", "aq", "ar", "as", "at",
        "au", "av", "aw", "ax", "ay", "az", "ba", "bb", "bc",
    ]
]


def SanitizeLabelFolder(label: str) -> str:
    collapsed = re.sub(r"\s+", "_", label.strip())
    return re.sub(r"[^A-Za-z0-9._-]", "_", collapsed)


def DeterministicSplit(key: str, val_fraction: float = 0.2) -> str:
    h = int(hashlib.sha256(key.encode()).hexdigest(), 16)
    return "val" if (h % 1000) < int(val_fraction * 1000) else "train"


def DownloadParts(download_dir: Path) -> None:
    download_dir.mkdir(parents=True, exist_ok=True)
    for url in SEAMAPD21_PARTS:
        filename = url.split("/")[-1]
        dest = download_dir / filename
        if dest.exists():
            print(f"  Already exists: {filename}")
            continue
        print(f"  Downloading {filename}...")
        subprocess.run(["curl", "-o", str(dest), "-L", url, "--progress-bar"], check=True)


def ExtractArchive(download_dir: Path, extract_dir: Path) -> None:
    extract_dir.mkdir(parents=True, exist_ok=True)
    MarkerFile = extract_dir / ".extraction_complete"
    if MarkerFile.exists():
        print("  Already extracted.")
        return
    print("  Joining and extracting tar parts...")
    parts = sorted(download_dir.glob("SEAMAPD21.tar.gz.*"))
    if not parts:
        print("  ERROR: No tar parts found")
        sys.exit(1)
    CatCmd = " ".join(f'"{p}"' for p in parts)
    subprocess.run(f'cat {CatCmd} | tar xzf - -C "{extract_dir}"', shell=True, check=True)
    MarkerFile.touch()


def ParseViameCsv(csv_path: Path) -> list[dict]:
    """Parse VIAME CSV format into detection records."""
    Detections = []
    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) < 11:
                continue
            try:
                Record = {
                    "track_id": parts[0].strip(),
                    "image_file": parts[1].strip(),
                    "frame_id": parts[2].strip(),
                    "tl_x": float(parts[3]),
                    "tl_y": float(parts[4]),
                    "br_x": float(parts[5]),
                    "br_y": float(parts[6]),
                    "confidence": float(parts[7]),
                    "species": parts[9].strip(),
                    "species_score": float(parts[10]),
                }
                if Record["species"] and Record["species_score"] > 0:
                    Detections.append(Record)
            except (ValueError, IndexError):
                continue
    return Detections


def IntegrateDetections(
    extract_dir: Path,
    output_dir: Path,
    min_box_size: int = 32,
    limit_per_species: int = 2000,
) -> None:
    """Crop bounding boxes from SEAMAPD21 images and save to training folder structure."""
    CsvFiles = list(extract_dir.rglob("*.csv"))
    print(f"  Found {len(CsvFiles)} CSV annotation files")

    SpeciesCounts: dict[str, int] = {}
    TotalCropped = 0

    for CsvFile in CsvFiles:
        Detections = ParseViameCsv(CsvFile)
        ImageDir = CsvFile.parent

        for det in Detections:
            Species = det["species"]
            FolderName = SanitizeLabelFolder(Species)
            if SpeciesCounts.get(FolderName, 0) >= limit_per_species:
                continue

            ImagePath = ImageDir / det["image_file"]
            if not ImagePath.exists():
                for parent in extract_dir.rglob(det["image_file"]):
                    ImagePath = parent
                    break
                else:
                    continue

            BoxW = det["br_x"] - det["tl_x"]
            BoxH = det["br_y"] - det["tl_y"]
            if BoxW < min_box_size or BoxH < min_box_size:
                continue

            CropKey = f"seamapd21_{det['track_id']}_{det['frame_id']}"
            Split = DeterministicSplit(CropKey)
            DestDir = output_dir / Split / FolderName
            DestDir.mkdir(parents=True, exist_ok=True)
            DestPath = DestDir / f"seamapd21_{det['track_id']}_{det['frame_id']}.jpg"

            if DestPath.exists():
                SpeciesCounts[FolderName] = SpeciesCounts.get(FolderName, 0) + 1
                continue

            try:
                with Image.open(ImagePath) as img:
                    Crop = img.crop((int(det["tl_x"]), int(det["tl_y"]), int(det["br_x"]), int(det["br_y"])))
                    Crop.convert("RGB").save(DestPath, format="JPEG", quality=95)
                    TotalCropped += 1
                    SpeciesCounts[FolderName] = SpeciesCounts.get(FolderName, 0) + 1
            except Exception:
                continue

        if TotalCropped % 1000 == 0 and TotalCropped > 0:
            print(f"    Cropped {TotalCropped:,} images across {len(SpeciesCounts)} species")

    print(f"\nDone: {TotalCropped:,} crops across {len(SpeciesCounts)} species")


def main() -> None:
    import argparse

    Parser = argparse.ArgumentParser(description="Download and integrate SEAMAPD21 dataset.")
    Parser.add_argument("--download-dir", type=Path, default=Path("D:/bathyal_training_data_seamapd21/parts"))
    Parser.add_argument("--extract-dir", type=Path, default=Path("D:/bathyal_training_data_seamapd21/extracted"))
    Parser.add_argument("--output-dir", type=Path, default=Path("D:/bathyal_training_data"))
    Parser.add_argument("--limit-per-species", type=int, default=2000)
    Parser.add_argument("--skip-download", action="store_true")
    Args = Parser.parse_args()

    if not Args.skip_download:
        print("Step 1: Downloading SEAMAPD21 parts (~26 GB)...")
        DownloadParts(Args.download_dir)
        print("Step 2: Extracting archive...")
        ExtractArchive(Args.download_dir, Args.extract_dir)
    else:
        print("Skipping download, using existing extraction.")

    print("Step 3: Cropping detections into training folders...")
    IntegrateDetections(Args.extract_dir, Args.output_dir, limit_per_species=Args.limit_per_species)


if __name__ == "__main__":
    main()
