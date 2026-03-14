"""iNaturalist data ingestion for building classifier-ready datasets."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from PIL import Image

from .ingest import (
    append_manifest_record,
    deterministic_split,
    download_file,
    file_extension_from_url,
    load_existing_manifest_state,
    sanitize_filename_component,
    sanitize_label_folder,
)

USER_AGENT = "bathyal-ai/0.1 (species-classifier; non-commercial research)"
API_BASE = "https://api.inaturalist.org/v1"


@dataclass(slots=True)
class INaturalistIngestConfig:
    concepts: list[str]
    output_dir: Path = Path("datasets")
    limit_per_concept: int = 100
    val_fraction: float = 0.2
    photo_size: str = "large"
    photo_license: str = "cc-by,cc-by-nc,cc-by-sa,cc-by-nc-sa,cc0"
    min_image_size: int = 224
    dataset_role: str = "known"
    cache_dir: Path = Path(".cache/ingest/images/inaturalist")
    api_delay: float = 1.0
    download_delay: float = 0.2


def _api_get(endpoint: str, params: dict[str, str | int] | None = None, max_retries: int = 3) -> dict:
    url = f"{API_BASE}/{endpoint}"
    if params:
        url = f"{url}?{urlencode(params, quote_via=quote)}"
    request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    for attempt in range(max_retries):
        try:
            with urlopen(request, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except (URLError, ConnectionResetError, OSError, HTTPError) as exc:
            if isinstance(exc, HTTPError) and exc.code == 429:
                time.sleep(10)
                continue
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)
    return {}


def resolve_taxon_id(concept: str) -> int | None:
    result = _api_get("taxa", {"q": concept, "rank": "species,genus,family,order", "is_active": "true", "per_page": "5"})
    for taxon in result.get("results", []):
        if taxon.get("name", "").lower() == concept.lower():
            return int(taxon["id"])
    candidates = result.get("results", [])
    if candidates:
        return int(candidates[0]["id"])
    return None


def _photo_url_for_size(url: str, size: str) -> str:
    return re.sub(r"/(square|thumb|small|medium|large|original)\.", f"/{size}.", url)


def manifest_path(output_dir: Path) -> Path:
    return output_dir / "manifests" / "inaturalist.jsonl"


def ingest_inaturalist(config: INaturalistIngestConfig) -> dict[str, object]:
    if not config.concepts:
        raise ValueError("At least one concept is required")
    if config.dataset_role not in {"known", "unknown"}:
        raise ValueError("dataset_role must be 'known' or 'unknown'")

    manifest = manifest_path(config.output_dir)
    seen_crop_keys, existing_image_splits, existing_concept_counts = load_existing_manifest_state(manifest)

    summary: dict[str, object] = {
        "source": "inaturalist",
        "output_dir": str(config.output_dir),
        "dataset_role": config.dataset_role,
        "concepts": config.concepts,
        "limit_per_concept": config.limit_per_concept,
        "concept_summaries": [],
    }
    total_downloaded = 0

    for query_concept in config.concepts:
        existing_count = existing_concept_counts.get(query_concept, 0)
        if existing_count >= config.limit_per_concept:
            summary["concept_summaries"].append({
                "query_concept": query_concept, "taxon_id": None,
                "observations_queried": 0, "photos_downloaded": 0, "skipped_existing": existing_count,
            })
            continue

        print(f"  Resolving taxon: {query_concept}...")
        taxon_id = resolve_taxon_id(query_concept)
        if taxon_id is None:
            print(f"    WARNING: Could not resolve taxon ID for '{query_concept}', skipping")
            summary["concept_summaries"].append({
                "query_concept": query_concept, "taxon_id": None,
                "observations_queried": 0, "photos_downloaded": 0, "error": "taxon_not_found",
            })
            continue
        time.sleep(config.api_delay)

        remaining = config.limit_per_concept - existing_count
        concept_downloaded = 0
        observations_queried = 0
        id_above = 0

        while concept_downloaded < remaining:
            params = {
                "taxon_id": str(taxon_id),
                "quality_grade": "research",
                "photos": "true",
                "photo_license": config.photo_license,
                "per_page": "200",
                "order": "asc",
                "order_by": "id",
                "id_above": str(id_above),
            }
            result = _api_get("observations", params)
            observations = result.get("results", [])
            if not observations:
                break
            time.sleep(config.api_delay)

            for obs in observations:
                if concept_downloaded >= remaining:
                    break

                observations_queried += 1
                obs_id = obs.get("id")
                taxon = obs.get("taxon", {})
                species_name = taxon.get("name", query_concept)
                photos = obs.get("photos", [])
                if not photos or not obs_id:
                    continue

                id_above = max(id_above, int(obs_id))

                photo = photos[0]
                photo_id = photo.get("id")
                photo_url = photo.get("url")
                if not photo_url or not photo_id:
                    continue

                crop_key = f"inaturalist:{obs_id}:{photo_id}"
                if crop_key in seen_crop_keys:
                    continue

                photo_url = _photo_url_for_size(photo_url, config.photo_size)
                extension = file_extension_from_url(photo_url)
                cache_path = config.cache_dir / f"inaturalist_{sanitize_filename_component(str(obs_id))}_{sanitize_filename_component(str(photo_id))}{extension}"

                try:
                    needs_download = not cache_path.exists()
                    image_path = download_file(photo_url, cache_path)
                    if needs_download:
                        time.sleep(config.download_delay)
                except (URLError, ConnectionResetError, OSError):
                    continue

                crop_label = species_name if config.dataset_role == "known" else "unknown"
                if config.dataset_role == "known":
                    split = existing_image_splits.get(
                        str(obs_id),
                        deterministic_split(str(obs_id), config.val_fraction),
                    )
                else:
                    split = "unknown_val"

                if split == "unknown_val":
                    destination_dir = config.output_dir / split
                else:
                    destination_dir = config.output_dir / split / sanitize_label_folder(crop_label)
                destination_dir.mkdir(parents=True, exist_ok=True)

                destination_name = f"inaturalist_{sanitize_filename_component(str(obs_id))}_{sanitize_filename_component(str(photo_id))}{extension}"
                destination = destination_dir / destination_name

                try:
                    with Image.open(image_path) as img:
                        width, height = img.size
                        if min(width, height) < config.min_image_size:
                            continue
                        if not destination.exists():
                            img.convert("RGB").save(destination, format="JPEG", quality=95)
                except Exception as exc:
                    print(f"    WARNING: Failed to process image {image_path}: {exc}")
                    continue

                record = {
                    "source": "inaturalist",
                    "dataset_role": config.dataset_role,
                    "query_concept": query_concept,
                    "label": crop_label,
                    "split": split,
                    "crop_path": str(destination),
                    "source_image_path": str(image_path),
                    "source_image_url": photo_url,
                    "observation_id": obs_id,
                    "photo_id": photo_id,
                    "species_name": species_name,
                    "taxon_id": taxon_id,
                    "photo_license": photo.get("license_code"),
                    "attribution": photo.get("attribution"),
                    "location": obs.get("location"),
                    "observed_on": obs.get("observed_on"),
                }
                append_manifest_record(manifest, record)
                seen_crop_keys.add(crop_key)
                if config.dataset_role == "known":
                    existing_image_splits.setdefault(str(obs_id), split)
                concept_downloaded += 1
                total_downloaded += 1

            if len(observations) < 200:
                break

        print(f"    {query_concept} (taxon {taxon_id}): {concept_downloaded} photos from {observations_queried} observations")
        summary["concept_summaries"].append({
            "query_concept": query_concept,
            "taxon_id": taxon_id,
            "observations_queried": observations_queried,
            "photos_downloaded": concept_downloaded,
        })

    summary["total_photos_downloaded"] = total_downloaded
    summary["manifest"] = str(manifest)
    return summary
