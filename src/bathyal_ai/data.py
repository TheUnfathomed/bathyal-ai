"""Dataset indexing helpers for Bathyal AI."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


@dataclass(slots=True)
class LabeledExample:
    path: Path
    label: str


@dataclass(slots=True)
class IndexedDataset:
    root: Path
    examples: list[LabeledExample]

    @property
    def labels(self) -> list[str]:
        return sorted({example.label for example in self.examples})

    @property
    def label_count(self) -> int:
        return len(self.labels)

    @property
    def example_count(self) -> int:
        return len(self.examples)

    @property
    def label_to_examples(self) -> dict[str, list[LabeledExample]]:
        grouped: dict[str, list[LabeledExample]] = defaultdict(list)
        for example in self.examples:
            grouped[example.label].append(example)
        return dict(grouped)


def list_images(root: Path) -> list[Path]:
    if root.is_file():
        if root.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image file: {root}")
        return [root]
    if not root.exists():
        raise FileNotFoundError(f"Path does not exist: {root}")
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def normalize_path(path: Path) -> Path:
    return path.resolve()


def format_label(raw_label: str) -> str:
    return raw_label.replace("_", " ").replace("-", " ").strip()


def infer_label_from_path(root: Path, path: Path) -> str:
    if root.is_file():
        return format_label(path.stem)
    relative = path.relative_to(root)
    if len(relative.parts) > 1:
        return format_label(relative.parts[0])
    return format_label(path.stem)


def index_labeled_dataset(root: Path, excluded_paths: Iterable[Path] | None = None) -> IndexedDataset:
    excluded = {normalize_path(path) for path in (excluded_paths or [])}
    image_paths = [path for path in list_images(root) if normalize_path(path) not in excluded]
    if not image_paths:
        raise ValueError(f"No images found under {root}")
    examples = [LabeledExample(path=path, label=infer_label_from_path(root, path)) for path in image_paths]
    return IndexedDataset(root=root, examples=examples)


def filter_dataset_by_min_examples(dataset: IndexedDataset, min_examples_per_label: int) -> IndexedDataset:
    min_examples = max(1, int(min_examples_per_label))
    kept_examples: list[LabeledExample] = []
    for label, examples in dataset.label_to_examples.items():
        if len(examples) >= min_examples:
            kept_examples.extend(examples)
    return IndexedDataset(root=dataset.root, examples=sorted(kept_examples, key=lambda example: str(example.path)))


def ensure_labels_subset(dataset: IndexedDataset, allowed_labels: Iterable[str], dataset_name: str) -> None:
    unknown_labels = sorted(set(dataset.labels) - set(allowed_labels))
    if unknown_labels:
        unknown_text = ", ".join(unknown_labels)
        raise ValueError(f"{dataset_name} contains labels that are not present in the classifier: {unknown_text}")


def fingerprint_examples(examples: Iterable[LabeledExample], extra: dict[str, object] | None = None) -> str:
    payload = []
    for example in sorted(examples, key=lambda item: str(item.path)):
        stat = example.path.stat()
        payload.append(
            {
                "path": str(example.path.resolve()),
                "label": example.label,
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    if extra:
        payload.append({"extra": extra})
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def infer_ground_truth_label(source_root: Path, path: Path) -> str:
    return infer_label_from_path(source_root, path)
