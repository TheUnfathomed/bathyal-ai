"""Data ingestion helpers for building classifier-ready crop datasets."""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from PIL import Image

from fathomnet import dto
from fathomnet.api import images, taxa


@dataclass(slots=True)
class FathomNetIngestConfig:
    concepts: list[str]
    output_dir: Path = Path("datasets")
    limit_per_concept: int = 100
    page_size: int = 100
    val_fraction: float = 0.2
    padding_fraction: float = 0.1
    square_crops: bool = True
    min_box_size: int = 32
    dataset_role: str = "known"
    cache_dir: Path = Path(".cache/ingest/images/fathomnet")
    include_descendants: bool = False
    taxa_provider: str | None = None
    review_states: list[str] = field(default_factory=lambda: ["VERIFIED"])
    min_depth: float | None = None
    max_depth: float | None = None
    imaging_types: list[str] = field(default_factory=list)


USER_AGENT = "bathyal-ai/0.1"


def sanitize_label_folder(label: str) -> str:
    collapsed = re.sub(r"\s+", "_", label.strip())
    return re.sub(r"[^A-Za-z0-9._-]", "_", collapsed)


def sanitize_filename_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", value)


def normalize_review_state(review_state: object) -> str | None:
    if review_state is None:
        return None
    if hasattr(review_state, "value"):
        return str(review_state.value)
    return str(review_state)


def deterministic_split(key: str, val_fraction: float) -> str:
    if val_fraction <= 0.0:
        return "train"
    if val_fraction >= 1.0:
        return "val"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:8]
    ratio = int(digest, 16) / 0xFFFFFFFF
    return "val" if ratio < val_fraction else "train"


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def file_extension_from_url(url: str) -> str:
    suffix = Path(urlparse(url).path).suffix.lower()
    if suffix in {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}:
        return suffix
    return ".jpg"


def download_file(url: str, destination: Path, max_retries: int = 3) -> Path:
    if destination.exists():
        return destination
    ensure_parent(destination)
    request = Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(max_retries):
        try:
            with urlopen(request, timeout=60) as response:
                destination.write_bytes(response.read())
            return destination
        except (URLError, ConnectionResetError, OSError):
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)
    return destination


def clamp_crop_box(
    *,
    x: int,
    y: int,
    width: int,
    height: int,
    image_width: int,
    image_height: int,
    padding_fraction: float,
    square: bool,
) -> tuple[int, int, int, int]:
    crop_width = width * (1.0 + 2.0 * padding_fraction)
    crop_height = height * (1.0 + 2.0 * padding_fraction)
    if square:
        size = max(crop_width, crop_height)
        crop_width = size
        crop_height = size

    center_x = x + width / 2.0
    center_y = y + height / 2.0

    left = int(math.floor(center_x - crop_width / 2.0))
    top = int(math.floor(center_y - crop_height / 2.0))
    right = int(math.ceil(center_x + crop_width / 2.0))
    bottom = int(math.ceil(center_y + crop_height / 2.0))

    left = max(0, left)
    top = max(0, top)
    right = min(image_width, max(left + 1, right))
    bottom = min(image_height, max(top + 1, bottom))
    return left, top, right, bottom


def concept_match_set(query_concept: str, include_descendants: bool, taxa_provider: str | None) -> set[str]:
    if not include_descendants:
        return {query_concept}
    if not taxa_provider:
        raise ValueError("taxa_provider is required when include_descendants is enabled")
    descendants = taxa.find_taxa(taxa_provider, query_concept)
    names = {entry.name for entry in descendants if entry.name}
    names.add(query_concept)
    return names


def build_constraints(config: FathomNetIngestConfig, concept: str) -> dto.GeoImageConstraints:
    return dto.GeoImageConstraints(
        concept=concept,
        taxaProviderName=config.taxa_provider if config.include_descendants else None,
        imagingTypes=config.imaging_types or None,
        includeVerified=True,
        includeUnverified=True,
        minDepth=config.min_depth,
        maxDepth=config.max_depth,
        limit=config.page_size,
        offset=0,
    )


def manifest_path(output_dir: Path) -> Path:
    return output_dir / "manifests" / "fathomnet.jsonl"


def append_manifest_record(path: Path, record: dict[str, object]) -> None:
    ensure_parent(path)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def crop_key_for_record(record: dict[str, object]) -> str | None:
    if record.get("box_uuid"):
        return str(record["box_uuid"])
    image_uuid = record.get("image_uuid")
    box_concept = record.get("box_concept") or record.get("label")
    bbox = record.get("bbox_xywh")
    if image_uuid and box_concept and isinstance(bbox, list) and len(bbox) == 4:
        return f"{image_uuid}:{box_concept}:{bbox[0]}:{bbox[1]}:{bbox[2]}:{bbox[3]}"
    return None


def load_existing_manifest_state(path: Path) -> tuple[set[str], dict[str, str], dict[str, int]]:
    existing_crop_keys: set[str] = set()
    existing_image_splits: dict[str, str] = {}
    concept_crop_counts: dict[str, int] = {}
    if not path.exists():
        return existing_crop_keys, existing_image_splits, concept_crop_counts

    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        crop_key = crop_key_for_record(record)
        if crop_key:
            existing_crop_keys.add(crop_key)
        image_uuid = record.get("image_uuid")
        split = record.get("split")
        if image_uuid and split in {"train", "val"} and image_uuid not in existing_image_splits:
            existing_image_splits[str(image_uuid)] = str(split)
        query_concept = record.get("query_concept")
        if query_concept:
            concept_crop_counts[query_concept] = concept_crop_counts.get(query_concept, 0) + 1
    return existing_crop_keys, existing_image_splits, concept_crop_counts


def ingest_fathomnet(config: FathomNetIngestConfig) -> dict[str, object]:
    if not config.concepts:
        raise ValueError("At least one concept is required")
    if config.dataset_role not in {"known", "unknown"}:
        raise ValueError("dataset_role must be 'known' or 'unknown'")

    requested_review_states = {state.strip().upper() for state in config.review_states if state.strip()}
    manifest = manifest_path(config.output_dir)
    seen_crop_keys, existing_image_splits, existing_concept_counts = load_existing_manifest_state(manifest)

    summary: dict[str, object] = {
        "source": "fathomnet",
        "output_dir": str(config.output_dir),
        "dataset_role": config.dataset_role,
        "concepts": config.concepts,
        "limit_per_concept": config.limit_per_concept,
        "padding_fraction": config.padding_fraction,
        "square_crops": config.square_crops,
        "concept_summaries": [],
    }

    total_crops = 0

    for query_concept in config.concepts:
        existing_count = existing_concept_counts.get(query_concept, 0)
        if existing_count >= config.limit_per_concept:
            summary["concept_summaries"].append(
                {"query_concept": query_concept, "matched_concepts": [query_concept],
                 "images_considered": 0, "crops_written": 0, "skipped_existing": existing_count}
            )
            continue

        matched_concepts = concept_match_set(
            query_concept=query_concept,
            include_descendants=config.include_descendants,
            taxa_provider=config.taxa_provider,
        )
        constraints = build_constraints(config, query_concept)
        concept_written = 0
        remaining = config.limit_per_concept - existing_count
        concept_images_seen = 0
        offset = 0

        while concept_written < remaining:
            constraints.offset = offset
            batch = images.find(constraints)
            if not batch:
                break

            for image_record in batch:
                if concept_written >= remaining:
                    break
                if not image_record.url or not image_record.boundingBoxes or not image_record.uuid:
                    continue

                concept_images_seen += 1
                cache_path = config.cache_dir / f"{sanitize_filename_component(image_record.uuid)}{file_extension_from_url(image_record.url)}"
                needs_download = not cache_path.exists()
                image_path = download_file(image_record.url, cache_path)
                if needs_download:
                    time.sleep(0.2)

                with Image.open(image_path) as source_image:
                    rgb_image = source_image.convert("RGB")
                    image_width, image_height = rgb_image.size

                    for box in image_record.boundingBoxes:
                        if concept_written >= remaining:
                            break
                        if not box.concept or box.concept not in matched_concepts:
                            continue
                        if box.x is None or box.y is None or box.width is None or box.height is None:
                            continue
                        if min(box.width, box.height) < config.min_box_size:
                            continue

                        review_state = normalize_review_state(box.reviewState)
                        if requested_review_states and (review_state or "").upper() not in requested_review_states:
                            continue

                        crop_key = box.uuid or f"{image_record.uuid}:{box.concept}:{box.x}:{box.y}:{box.width}:{box.height}"
                        if crop_key in seen_crop_keys:
                            continue

                        left, top, right, bottom = clamp_crop_box(
                            x=int(box.x),
                            y=int(box.y),
                            width=int(box.width),
                            height=int(box.height),
                            image_width=image_width,
                            image_height=image_height,
                            padding_fraction=config.padding_fraction,
                            square=config.square_crops,
                        )
                        crop = rgb_image.crop((left, top, right, bottom))

                        crop_label = box.concept if config.dataset_role == "known" else "unknown"
                        if config.dataset_role == "known":
                            split = existing_image_splits.get(
                                image_record.uuid,
                                deterministic_split(image_record.uuid, config.val_fraction),
                            )
                        else:
                            split = "unknown_val"

                        if split == "unknown_val":
                            destination_dir = config.output_dir / split
                        else:
                            destination_dir = config.output_dir / split / sanitize_label_folder(crop_label)
                        destination_dir.mkdir(parents=True, exist_ok=True)

                        crop_name = (
                            f"fathomnet_{sanitize_filename_component(image_record.uuid)}_"
                            f"{sanitize_filename_component(box.uuid or str(concept_written))}.jpg"
                        )
                        destination = destination_dir / crop_name
                        if destination.exists():
                            seen_crop_keys.add(crop_key)
                            if config.dataset_role == "known":
                                existing_image_splits.setdefault(image_record.uuid, split)
                            continue

                        crop.save(destination, format="JPEG", quality=95)

                        record = {
                            "source": "fathomnet",
                            "dataset_role": config.dataset_role,
                            "query_concept": query_concept,
                            "matched_concepts": sorted(matched_concepts),
                            "label": crop_label,
                            "split": split,
                            "crop_path": str(destination),
                            "source_image_path": str(image_path),
                            "source_image_url": image_record.url,
                            "image_uuid": image_record.uuid,
                            "box_uuid": box.uuid,
                            "box_concept": box.concept,
                            "box_review_state": review_state,
                            "bbox_xywh": [int(box.x), int(box.y), int(box.width), int(box.height)],
                            "crop_xyxy": [left, top, right, bottom],
                            "image_width": image_record.width,
                            "image_height": image_record.height,
                            "depth_meters": image_record.depthMeters,
                            "latitude": image_record.latitude,
                            "longitude": image_record.longitude,
                            "contributors_email": image_record.contributorsEmail,
                            "timestamp": image_record.timestamp,
                        }
                        append_manifest_record(manifest, record)
                        seen_crop_keys.add(crop_key)
                        if config.dataset_role == "known":
                            existing_image_splits.setdefault(image_record.uuid, split)
                        concept_written += 1
                        total_crops += 1

            if len(batch) < config.page_size:
                break
            offset += config.page_size

        summary["concept_summaries"].append(
            {
                "query_concept": query_concept,
                "matched_concepts": sorted(matched_concepts),
                "images_considered": concept_images_seen,
                "crops_written": concept_written,
            }
        )

    summary["total_crops_written"] = total_crops
    summary["manifest"] = str(manifest)
    return summary
