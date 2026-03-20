"""Bulk-download iNaturalist images via AWS S3 open data, filtered by species list."""

from __future__ import annotations

import csv
import gzip
import hashlib
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

S3_PHOTO_BASE = "https://inaturalist-open-data.s3.amazonaws.com/photos"
USER_AGENT = "bathyal-ai/0.1 (species-classifier; non-commercial research)"
DEFAULT_SIZE = "large"


def SanitizeLabelFolder(label: str) -> str:
    collapsed = re.sub(r"\s+", "_", label.strip())
    return re.sub(r"[^A-Za-z0-9._-]", "_", collapsed)


def DeterministicSplit(key: str, val_fraction: float = 0.2) -> str:
    h = int(hashlib.sha256(key.encode()).hexdigest(), 16)
    return "val" if (h % 1000) < int(val_fraction * 1000) else "train"


def LoadTargetTaxonIds(taxa_path: Path, species_names: set[str]) -> dict[int, str]:
    """Map taxon_id -> species name for all matching species in taxa.csv."""
    NormalizedLookup = {name.lower().strip(): name for name in species_names}
    TaxonMap: dict[int, str] = {}

    with gzip.open(taxa_path, "rt", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row.get("rank") != "species":
                continue
            name = row.get("name", "").strip()
            if name.lower() in NormalizedLookup:
                TaxonMap[int(row["taxon_id"])] = name

    print(f"  Matched {len(TaxonMap)} taxon IDs from taxa.csv")
    return TaxonMap


def StreamMatchingObservations(obs_path: Path, target_taxon_ids: set[int]) -> dict[str, int]:
    """Stream observations.csv.gz, return {observation_uuid: taxon_id} for matches."""
    Matches: dict[str, int] = {}
    Processed = 0

    with gzip.open(obs_path, "rt", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            Processed += 1
            if Processed % 5_000_000 == 0:
                print(f"    ...processed {Processed:,} observations, {len(Matches):,} matches")
            if row.get("quality_grade") != "research":
                continue
            try:
                TaxonId = int(row["taxon_id"])
            except (ValueError, KeyError):
                continue
            if TaxonId in target_taxon_ids:
                Matches[row["observation_uuid"]] = TaxonId

    print(f"  Found {len(Matches):,} matching observations from {Processed:,} total")
    return Matches


def StreamMatchingPhotos(
    photos_path: Path,
    observation_map: dict[str, int],
    taxon_names: dict[int, str],
    limit_per_species: int,
) -> list[dict]:
    """Stream photos.csv.gz, return photo records for matching observations."""
    SpeciesCounts: dict[str, int] = {}
    Records: list[dict] = []
    Processed = 0

    with gzip.open(photos_path, "rt", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            Processed += 1
            if Processed % 10_000_000 == 0:
                print(f"    ...processed {Processed:,} photos, {len(Records):,} matched")
            ObsUuid = row.get("observation_uuid")
            if ObsUuid not in observation_map:
                continue
            TaxonId = observation_map[ObsUuid]
            SpeciesName = taxon_names[TaxonId]
            FolderName = SanitizeLabelFolder(SpeciesName)

            CurrentCount = SpeciesCounts.get(FolderName, 0)
            if CurrentCount >= limit_per_species:
                continue

            PhotoId = row.get("photo_id", "")
            Ext = row.get("extension", "jpg")
            if not PhotoId:
                continue

            Records.append({
                "photo_id": PhotoId,
                "extension": Ext,
                "species": SpeciesName,
                "folder": FolderName,
                "obs_uuid": ObsUuid,
            })
            SpeciesCounts[FolderName] = CurrentCount + 1

    UniqueSpecies = len(SpeciesCounts)
    print(f"  Selected {len(Records):,} photos across {UniqueSpecies} species from {Processed:,} total")
    return Records


def DownloadPhoto(record: dict, output_dir: Path, size: str) -> str | None:
    PhotoId = record["photo_id"]
    Ext = record["extension"]
    Folder = record["folder"]
    Split = DeterministicSplit(record["obs_uuid"])

    DestDir = output_dir / Split / Folder
    DestDir.mkdir(parents=True, exist_ok=True)
    DestPath = DestDir / f"inat_s3_{PhotoId}.{Ext}"

    if DestPath.exists():
        return None

    Url = f"{S3_PHOTO_BASE}/{PhotoId}/{size}.{Ext}"
    Req = Request(Url, headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(Req, timeout=30) as resp:
            DestPath.write_bytes(resp.read())
        return str(DestPath)
    except (HTTPError, URLError, OSError):
        return None


def main() -> None:
    import argparse

    Parser = argparse.ArgumentParser(description="Bulk download iNat images via S3 open data.")
    Parser.add_argument("--taxa-csv", type=Path, required=True)
    Parser.add_argument("--observations-csv", type=Path, required=True)
    Parser.add_argument("--photos-csv", type=Path, required=True)
    Parser.add_argument("--species-file", type=Path, required=True, help="Newline-delimited species names.")
    Parser.add_argument("--output-dir", type=Path, required=True)
    Parser.add_argument("--limit-per-species", type=int, default=500)
    Parser.add_argument("--photo-size", type=str, default=DEFAULT_SIZE)
    Parser.add_argument("--workers", type=int, default=16)
    Args = Parser.parse_args()

    SpeciesNames = set()
    for line in Args.species_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            SpeciesNames.add(line)
    # Also add species from existing train folders
    TrainDir = Args.output_dir / "train"
    if TrainDir.exists():
        for d in TrainDir.iterdir():
            if d.is_dir():
                SpeciesNames.add(d.name.replace("_", " "))
    print(f"Targeting {len(SpeciesNames)} species")

    print("Step 1: Loading taxa...")
    TaxonMap = LoadTargetTaxonIds(Args.taxa_csv, SpeciesNames)
    if not TaxonMap:
        print("ERROR: No matching taxa found")
        sys.exit(1)

    print("Step 2: Streaming observations (this takes a while)...")
    ObsMap = StreamMatchingObservations(Args.observations_csv, set(TaxonMap.keys()))

    print("Step 3: Streaming photos...")
    PhotoRecords = StreamMatchingPhotos(Args.photos_csv, ObsMap, TaxonMap, Args.limit_per_species)

    # Count existing images per species
    ExistingCounts: dict[str, int] = {}
    if TrainDir.exists():
        for d in TrainDir.iterdir():
            if d.is_dir():
                ExistingCounts[d.name] = len(list(d.iterdir()))
    ValDir = Args.output_dir / "val"
    if ValDir.exists():
        for d in ValDir.iterdir():
            if d.is_dir():
                ExistingCounts[d.name] = ExistingCounts.get(d.name, 0) + len(list(d.iterdir()))

    # Filter out species that already have enough
    Filtered = []
    FilteredCounts: dict[str, int] = {}
    for rec in PhotoRecords:
        Folder = rec["folder"]
        Existing = ExistingCounts.get(Folder, 0)
        Added = FilteredCounts.get(Folder, 0)
        if Existing + Added < Args.limit_per_species:
            Filtered.append(rec)
            FilteredCounts[Folder] = Added + 1
    print(f"  After accounting for existing data: {len(Filtered):,} photos to download")

    if not Filtered:
        print("Nothing to download.")
        return

    print(f"Step 4: Downloading {len(Filtered):,} photos with {Args.workers} workers...")
    Downloaded = 0
    Skipped = 0
    Failed = 0

    with ThreadPoolExecutor(max_workers=Args.workers) as pool:
        Futures = {pool.submit(DownloadPhoto, rec, Args.output_dir, Args.photo_size): rec for rec in Filtered}
        for future in as_completed(Futures):
            result = future.result()
            if result is None:
                Skipped += 1
            else:
                Downloaded += 1
            Total = Downloaded + Skipped + Failed
            if Total % 1000 == 0:
                print(f"    Progress: {Total:,}/{len(Filtered):,} (downloaded: {Downloaded:,})")

    print(f"\nDone: {Downloaded:,} downloaded, {Skipped:,} skipped/existing, {Failed:,} failed")


if __name__ == "__main__":
    main()
