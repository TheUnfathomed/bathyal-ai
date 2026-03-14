"""Production classifier bundle built on BioCLIP 2 embeddings."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from .embeddings import BioClipEmbedder


@dataclass(slots=True)
class ThresholdConfig:
    probability: float = 0.8
    margin: float = 0.05
    centroid_similarity: float = 0.6

    def to_dict(self) -> dict[str, float]:
        return {
            "probability": float(self.probability),
            "margin": float(self.margin),
            "centroid_similarity": float(self.centroid_similarity),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "ThresholdConfig":
        return cls(
            probability=float(payload.get("probability", 0.8)),
            margin=float(payload.get("margin", 0.05)),
            centroid_similarity=float(payload.get("centroid_similarity", 0.6)),
        )


class LinearClassifierHead(nn.Module):
    def __init__(self, feature_dim: int, class_count: int) -> None:
        super().__init__()
        self.linear = nn.Linear(feature_dim, class_count)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.linear(embeddings)


class SpeciesClassifierBundle:
    def __init__(
        self,
        *,
        bioclip_model: str,
        labels: Sequence[str],
        head: LinearClassifierHead,
        centroids: np.ndarray,
        temperature: float = 1.0,
        thresholds: ThresholdConfig | None = None,
        device: str = "auto",
        embed_batch_size: int = 16,
        top_candidate_count: int = 5,
        training_summary: dict[str, object] | None = None,
        created_at: str | None = None,
        bundle_dir: Path | None = None,
        lora_config: dict[str, object] | None = None,
    ) -> None:
        if centroids.ndim != 2:
            raise ValueError("Centroids must be a 2D matrix")
        if len(labels) != int(centroids.shape[0]):
            raise ValueError("Number of labels must match centroid count")

        self.bioclip_model = bioclip_model
        self.labels = list(labels)
        self.label_to_index = {label: index for index, label in enumerate(self.labels)}
        self.feature_dim = int(centroids.shape[1])
        self.top_candidate_count = max(1, int(top_candidate_count))
        self.thresholds = thresholds or ThresholdConfig()
        self.temperature = max(float(temperature), 1e-3)
        self.training_summary = training_summary or {}
        self.created_at = created_at or datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.bundle_dir = bundle_dir
        self.lora_config = lora_config

        self.embedder = BioClipEmbedder(model_name=bioclip_model, device=device, batch_size=embed_batch_size)
        self.device = self.embedder.device
        self.head = head.to(self.device)
        self.head.eval()
        self.centroids = self._normalize_embeddings(np.asarray(centroids, dtype=np.float32))

    @staticmethod
    def _normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim == 1:
            embeddings = embeddings[None, :]
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.clip(norms, a_min=1e-12, a_max=None)
        return embeddings / norms

    def predict_arrays(self, embeddings: np.ndarray) -> dict[str, np.ndarray]:
        normalized_embeddings = self._normalize_embeddings(embeddings)
        tensor = torch.from_numpy(normalized_embeddings).to(self.device)
        with torch.inference_mode():
            logits = self.head(tensor)
            probabilities = F.softmax(logits / self.temperature, dim=-1).cpu().numpy().astype(np.float32)

        ranked_indices = np.argsort(-probabilities, axis=1)
        top_indices = ranked_indices[:, 0].astype(np.int64)
        if probabilities.shape[1] > 1:
            second_indices = ranked_indices[:, 1].astype(np.int64)
            second_probabilities = probabilities[np.arange(len(probabilities)), second_indices]
        else:
            second_indices = np.full(len(probabilities), -1, dtype=np.int64)
            second_probabilities = np.zeros(len(probabilities), dtype=np.float32)

        top_probabilities = probabilities[np.arange(len(probabilities)), top_indices]
        centroid_similarities = normalized_embeddings @ self.centroids.T
        predicted_centroid_similarities = centroid_similarities[np.arange(len(centroid_similarities)), top_indices]

        return {
            "probabilities": probabilities,
            "ranked_indices": ranked_indices.astype(np.int64),
            "top_indices": top_indices,
            "second_indices": second_indices,
            "top_probabilities": top_probabilities.astype(np.float32),
            "second_probabilities": second_probabilities.astype(np.float32),
            "margins": (top_probabilities - second_probabilities).astype(np.float32),
            "centroid_similarities": centroid_similarities.astype(np.float32),
            "predicted_centroid_similarities": predicted_centroid_similarities.astype(np.float32),
        }

    def score_embedding(self, embedding: np.ndarray, top_candidate_count: int | None = None) -> dict[str, object]:
        score_arrays = self.predict_arrays(np.asarray(embedding, dtype=np.float32))
        candidate_count = self.top_candidate_count if top_candidate_count is None else max(1, int(top_candidate_count))
        candidate_indices = score_arrays["ranked_indices"][0, :candidate_count]
        candidates: list[dict[str, object]] = []
        for class_index in candidate_indices:
            candidates.append(
                {
                    "label": self.labels[int(class_index)],
                    "probability": round(float(score_arrays["probabilities"][0, int(class_index)]), 6),
                    "centroid_similarity": round(float(score_arrays["centroid_similarities"][0, int(class_index)]), 6),
                }
            )

        top_index = int(score_arrays["top_indices"][0])
        second_index = int(score_arrays["second_indices"][0])
        return {
            "top_label": self.labels[top_index],
            "top_probability": round(float(score_arrays["top_probabilities"][0]), 6),
            "second_best_label": self.labels[second_index] if second_index >= 0 else None,
            "second_best_probability": round(float(score_arrays["second_probabilities"][0]), 6),
            "margin": round(float(score_arrays["margins"][0]), 6),
            "centroid_similarity": round(float(score_arrays["predicted_centroid_similarities"][0]), 6),
            "candidate_rankings": candidates,
        }

    def decide(self, score: dict[str, object], thresholds: ThresholdConfig | None = None) -> dict[str, object]:
        active_thresholds = thresholds or self.thresholds
        top_probability = float(score["top_probability"])
        margin = float(score["margin"])
        centroid_similarity = float(score["centroid_similarity"])

        accepted = (
            top_probability >= active_thresholds.probability
            and margin >= active_thresholds.margin
            and centroid_similarity >= active_thresholds.centroid_similarity
        )

        rejection_reasons: list[str] = []
        if top_probability < active_thresholds.probability:
            rejection_reasons.append("probability")
        if margin < active_thresholds.margin:
            rejection_reasons.append("margin")
        if centroid_similarity < active_thresholds.centroid_similarity:
            rejection_reasons.append("centroid_similarity")

        return {
            "predicted_label": str(score["top_label"]) if accepted else "unknown",
            "accepted": accepted,
            "thresholds": active_thresholds.to_dict(),
            "rejection_reasons": rejection_reasons,
        }

    def classify(
        self,
        image: Image.Image,
        thresholds: ThresholdConfig | None = None,
        top_candidate_count: int | None = None,
    ) -> dict[str, object]:
        embedding = self.embedder.embed_images([image.convert("RGB")])[0]
        score = self.score_embedding(embedding, top_candidate_count=top_candidate_count)
        decision = self.decide(score, thresholds=thresholds)
        return {**score, **decision}

    def save(self, bundle_dir: Path) -> Path:
        bundle_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.head.state_dict()}, bundle_dir / "head.pt")
        np.save(bundle_dir / "centroids.npy", self.centroids.astype(np.float32))

        metadata = {
            "bundle_format_version": 2 if self.lora_config else 1,
            "bioclip_model": self.bioclip_model,
            "labels": self.labels,
            "feature_dim": self.feature_dim,
            "temperature": self.temperature,
            "thresholds": self.thresholds.to_dict(),
            "top_candidate_count": self.top_candidate_count,
            "training_summary": self.training_summary,
            "created_at": self.created_at,
            "lora_config": self.lora_config,
        }
        (bundle_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        self.bundle_dir = bundle_dir
        return bundle_dir

    @classmethod
    def load(
        cls,
        bundle_dir: Path,
        *,
        device: str = "auto",
        embed_batch_size: int = 16,
    ) -> "SpeciesClassifierBundle":
        metadata_path = bundle_dir / "metadata.json"
        centroids_path = bundle_dir / "centroids.npy"
        head_path = bundle_dir / "head.pt"

        if not metadata_path.exists() or not centroids_path.exists() or not head_path.exists():
            raise FileNotFoundError(f"Classifier bundle is incomplete: {bundle_dir}")

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        labels = [str(label) for label in metadata["labels"]]
        feature_dim = int(metadata["feature_dim"])
        head = LinearClassifierHead(feature_dim=feature_dim, class_count=len(labels))
        state = torch.load(head_path, map_location="cpu")
        head.load_state_dict(state["state_dict"])

        lora_config = metadata.get("lora_config")
        instance = cls(
            bioclip_model=str(metadata["bioclip_model"]),
            labels=labels,
            head=head,
            centroids=np.load(centroids_path).astype(np.float32),
            temperature=float(metadata.get("temperature", 1.0)),
            thresholds=ThresholdConfig.from_dict(metadata.get("thresholds", {})),
            device=device,
            embed_batch_size=embed_batch_size,
            top_candidate_count=int(metadata.get("top_candidate_count", 5)),
            training_summary=metadata.get("training_summary", {}),
            created_at=metadata.get("created_at"),
            bundle_dir=bundle_dir,
            lora_config=lora_config,
        )

        lora_weights_path = bundle_dir / "lora_weights.pt"
        if lora_config and lora_weights_path.exists():
            from .lora import LoraConfig, apply_lora_to_vision_encoder, load_lora_state_dict, merge_lora_weights

            lora_cfg = LoraConfig(
                rank=int(lora_config.get("rank", 8)),
                alpha=float(lora_config.get("alpha", 16.0)),
                target_blocks=lora_config.get("target_blocks"),
                targets=lora_config.get("targets", ["q", "k", "v", "o"]),
            )
            apply_lora_to_vision_encoder(instance.embedder.model, lora_cfg)
            state_dict = torch.load(lora_weights_path, map_location="cpu")
            load_lora_state_dict(instance.embedder.model, state_dict)
            merge_lora_weights(instance.embedder.model)

        return instance
