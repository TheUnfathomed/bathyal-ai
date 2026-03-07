"""BioCLIP embedding utilities with optional on-disk caching."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from PIL import Image

from .data import LabeledExample, fingerprint_examples


@dataclass(slots=True)
class EmbeddingSet:
    embeddings: np.ndarray
    labels: list[str]
    paths: list[str]
    fingerprint: str

    @property
    def feature_dim(self) -> int:
        if self.embeddings.ndim != 2:
            return 0
        return int(self.embeddings.shape[1])


class BioClipEmbedder:
    def __init__(self, model_name: str, device: str = "auto", batch_size: int = 16) -> None:
        self.model_name = model_name
        self.device = self._resolve_device(device)
        self.batch_size = batch_size
        self.model, self.preprocess = self._load_model()

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _load_model(self):
        try:
            model, preprocess = open_clip.create_model_from_pretrained(self.model_name)
        except Exception:
            model, _, preprocess = open_clip.create_model_and_transforms(self.model_name)
        model = model.to(self.device)
        model.eval()
        return model, preprocess

    def embed_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        if not images:
            raise ValueError("No images were provided for embedding")

        all_embeddings: list[np.ndarray] = []
        for start in range(0, len(images), self.batch_size):
            batch = images[start : start + self.batch_size]
            tensor = torch.stack([self.preprocess(image) for image in batch], dim=0).to(self.device)
            with torch.inference_mode():
                embeddings = self.model.encode_image(tensor)
                embeddings = F.normalize(embeddings, dim=-1)
            all_embeddings.append(embeddings.cpu().numpy().astype(np.float32))
        return np.concatenate(all_embeddings, axis=0)

    def embed_paths(self, paths: Sequence[Path]) -> np.ndarray:
        loaded_images: list[Image.Image] = []
        for path in paths:
            with Image.open(path) as image:
                loaded_images.append(image.convert("RGB"))
        return self.embed_images(loaded_images)

    def embed_examples(
        self,
        examples: Sequence[LabeledExample],
        cache_dir: Path | None = None,
        cache_namespace: str = "default",
    ) -> EmbeddingSet:
        if not examples:
            raise ValueError("No labeled examples were provided for embedding")

        fingerprint = fingerprint_examples(
            examples,
            extra={
                "model_name": self.model_name,
                "cache_namespace": cache_namespace,
            },
        )
        cache_path = None
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = cache_dir / f"{fingerprint}.npz"
            if cache_path.exists():
                cached = np.load(cache_path, allow_pickle=False)
                return EmbeddingSet(
                    embeddings=cached["embeddings"].astype(np.float32),
                    labels=cached["labels"].astype(str).tolist(),
                    paths=cached["paths"].astype(str).tolist(),
                    fingerprint=fingerprint,
                )

        embeddings = self.embed_paths([example.path for example in examples])
        labels = [example.label for example in examples]
        paths = [str(example.path) for example in examples]

        if cache_path is not None:
            np.savez_compressed(
                cache_path,
                embeddings=embeddings,
                labels=np.asarray(labels, dtype=str),
                paths=np.asarray(paths, dtype=str),
            )

        return EmbeddingSet(
            embeddings=embeddings,
            labels=labels,
            paths=paths,
            fingerprint=fingerprint,
        )

    def embed_unlabeled_paths(
        self,
        paths: Sequence[Path],
        cache_dir: Path | None = None,
        cache_namespace: str = "unlabeled",
    ) -> EmbeddingSet:
        examples = [LabeledExample(path=path, label="__unlabeled__") for path in paths]
        return self.embed_examples(examples=examples, cache_dir=cache_dir, cache_namespace=cache_namespace)
