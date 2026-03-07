"""Training flow for the production Bathyal AI classifier."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .classifier import LinearClassifierHead, SpeciesClassifierBundle, ThresholdConfig
from .data import ensure_labels_subset, filter_dataset_by_min_examples, index_labeled_dataset, list_images
from .embeddings import BioClipEmbedder
from .metrics import summarize_thresholds


@dataclass(slots=True)
class TrainingConfig:
    train_dir: Path
    val_dir: Path
    output_dir: Path = Path("artifacts/species_classifier")
    unknown_dir: Path | None = None
    bioclip_model: str = "hf-hub:imageomics/bioclip-2"
    embed_batch_size: int = 16
    classifier_batch_size: int = 64
    epochs: int = 200
    learning_rate: float = 5e-3
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05
    min_examples_per_label: int = 2
    target_precision: float = 0.98
    probability_thresholds: list[float] = field(
        default_factory=lambda: [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    )
    margin_thresholds: list[float] = field(default_factory=lambda: [0.0, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.25])
    centroid_thresholds: list[float] = field(default_factory=lambda: [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    device: str = "auto"
    cache_dir: Path | None = Path(".cache/embeddings")
    top_candidate_count: int = 5
    seed: int = 13
    early_stopping_patience: int = 30


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def encode_labels(labels: list[str], label_to_index: dict[str, int]) -> np.ndarray:
    return np.asarray([label_to_index[label] for label in labels], dtype=np.int64)


def compute_class_weights(targets: np.ndarray, class_count: int) -> torch.Tensor:
    counts = np.bincount(targets, minlength=class_count).astype(np.float32)
    counts[counts == 0.0] = 1.0
    weights = counts.sum() / counts
    weights = weights / weights.mean()
    return torch.from_numpy(weights.astype(np.float32))


def train_classifier_head(
    train_embeddings: np.ndarray,
    train_targets: np.ndarray,
    val_embeddings: np.ndarray,
    val_targets: np.ndarray,
    class_count: int,
    config: TrainingConfig,
    device: str,
) -> tuple[LinearClassifierHead, dict[str, float]]:
    feature_dim = int(train_embeddings.shape[1])
    head = LinearClassifierHead(feature_dim=feature_dim, class_count=class_count).to(device)

    class_weights = compute_class_weights(train_targets, class_count).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights, label_smoothing=config.label_smoothing)
    optimizer = torch.optim.AdamW(head.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    train_tensor = torch.from_numpy(train_embeddings.astype(np.float32)).to(device)
    train_target_tensor = torch.from_numpy(train_targets.astype(np.int64)).to(device)
    val_tensor = torch.from_numpy(val_embeddings.astype(np.float32)).to(device)
    val_target_tensor = torch.from_numpy(val_targets.astype(np.int64)).to(device)

    best_state = copy.deepcopy(head.state_dict())
    best_val_loss = float("inf")
    best_val_accuracy = 0.0
    epochs_without_improvement = 0

    for epoch in range(config.epochs):
        head.train()
        permutation = torch.randperm(len(train_target_tensor), device=device)
        batch_losses: list[float] = []

        for start in range(0, len(permutation), config.classifier_batch_size):
            batch_indices = permutation[start : start + config.classifier_batch_size]
            batch_embeddings = train_tensor[batch_indices]
            batch_targets = train_target_tensor[batch_indices]

            optimizer.zero_grad(set_to_none=True)
            logits = head(batch_embeddings)
            loss = criterion(logits, batch_targets)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.item()))

        head.eval()
        with torch.inference_mode():
            val_logits = head(val_tensor)
            val_loss = F.cross_entropy(val_logits, val_target_tensor, weight=class_weights).item()
            val_predictions = val_logits.argmax(dim=1)
            val_accuracy = float((val_predictions == val_target_tensor).float().mean().item())

        if val_loss + 1e-6 < best_val_loss:
            best_val_loss = val_loss
            best_val_accuracy = val_accuracy
            best_state = copy.deepcopy(head.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.early_stopping_patience:
                break

    head.load_state_dict(best_state)
    head.eval()
    return head, {
        "best_val_loss": round(best_val_loss, 6),
        "best_val_accuracy": round(best_val_accuracy, 6),
        "epochs_trained": epoch + 1,
    }


def fit_temperature(head: LinearClassifierHead, val_embeddings: np.ndarray, val_targets: np.ndarray, device: str) -> float:
    if len(val_targets) == 0:
        return 1.0

    head.eval()
    with torch.no_grad():
        logits = head(torch.from_numpy(val_embeddings.astype(np.float32)).to(device)).detach()
    labels = torch.from_numpy(val_targets.astype(np.int64)).to(device)

    log_temperature = torch.nn.Parameter(torch.zeros(1, device=device))
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=50, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        temperature = log_temperature.exp().clamp(min=1e-3, max=100.0)
        loss = F.cross_entropy(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.exp().detach().cpu().item())


def build_centroids(train_embeddings: np.ndarray, train_targets: np.ndarray, class_count: int) -> np.ndarray:
    centroids: list[np.ndarray] = []
    for class_index in range(class_count):
        class_embeddings = train_embeddings[train_targets == class_index]
        if len(class_embeddings) == 0:
            raise ValueError(f"Class index {class_index} has no training embeddings")
        centroid = class_embeddings.mean(axis=0)
        norm = max(float(np.linalg.norm(centroid)), 1e-12)
        centroids.append((centroid / norm).astype(np.float32))
    return np.vstack(centroids)


def rank_threshold_summary(summary: dict[str, object], target_precision: float) -> tuple[float, float, float, float, float]:
    precision = summary["precision_when_accepted"]
    precision_value = float(precision) if precision is not None else -1.0
    known_coverage = float(summary["known_coverage"])
    unknown_accept_rate = float(summary["unknown_accept_rate"])
    wrong_accept_rate_known = float(summary["wrong_accept_rate_known"])

    if precision is not None and precision_value >= target_precision:
        return (1.0, known_coverage, -unknown_accept_rate, -wrong_accept_rate_known, precision_value)
    return (0.0, precision_value, known_coverage, -unknown_accept_rate, -wrong_accept_rate_known)


def select_thresholds(
    bundle: SpeciesClassifierBundle,
    val_embeddings: np.ndarray,
    val_targets: np.ndarray,
    unknown_embeddings: np.ndarray | None,
    config: TrainingConfig,
) -> tuple[ThresholdConfig, dict[str, object], list[dict[str, object]]]:
    known_score_arrays = bundle.predict_arrays(val_embeddings)
    unknown_score_arrays = bundle.predict_arrays(unknown_embeddings) if unknown_embeddings is not None and len(unknown_embeddings) else None

    summaries: list[dict[str, object]] = []
    for probability_threshold in config.probability_thresholds:
        for margin_threshold in config.margin_thresholds:
            for centroid_threshold in config.centroid_thresholds:
                thresholds = ThresholdConfig(
                    probability=probability_threshold,
                    margin=margin_threshold,
                    centroid_similarity=centroid_threshold,
                )
                summary = summarize_thresholds(
                    known_score_arrays=known_score_arrays,
                    known_targets=val_targets,
                    thresholds=thresholds,
                    unknown_score_arrays=unknown_score_arrays,
                )
                summaries.append(summary)

    best_summary = max(summaries, key=lambda summary: rank_threshold_summary(summary, config.target_precision))
    thresholds = ThresholdConfig(
        probability=float(best_summary["probability_threshold"]),
        margin=float(best_summary["margin_threshold"]),
        centroid_similarity=float(best_summary["centroid_threshold"]),
    )

    ranked_summaries = sorted(
        summaries,
        key=lambda summary: rank_threshold_summary(summary, config.target_precision),
        reverse=True,
    )
    return thresholds, best_summary, ranked_summaries[:25]


def train_classifier(config: TrainingConfig) -> dict[str, object]:
    set_seed(config.seed)

    raw_train_dataset = index_labeled_dataset(config.train_dir)
    train_dataset = filter_dataset_by_min_examples(raw_train_dataset, config.min_examples_per_label)
    if train_dataset.example_count == 0:
        raise ValueError(
            f"No training labels remain after applying min_examples_per_label={config.min_examples_per_label}"
        )
    dropped_train_labels = sorted(set(raw_train_dataset.labels) - set(train_dataset.labels))

    val_dataset = index_labeled_dataset(config.val_dir)
    ensure_labels_subset(val_dataset, train_dataset.labels, "Validation dataset")

    unknown_paths = list_images(config.unknown_dir) if config.unknown_dir is not None else []

    embedder = BioClipEmbedder(
        model_name=config.bioclip_model,
        device=config.device,
        batch_size=config.embed_batch_size,
    )

    train_embedding_set = embedder.embed_examples(
        train_dataset.examples,
        cache_dir=config.cache_dir,
        cache_namespace="train",
    )
    val_embedding_set = embedder.embed_examples(
        val_dataset.examples,
        cache_dir=config.cache_dir,
        cache_namespace="val",
    )

    unknown_embeddings = None
    unknown_paths_text: list[str] = []
    if unknown_paths:
        unknown_embedding_set = embedder.embed_unlabeled_paths(
            unknown_paths,
            cache_dir=config.cache_dir,
            cache_namespace="unknown",
        )
        unknown_embeddings = unknown_embedding_set.embeddings
        unknown_paths_text = unknown_embedding_set.paths

    labels = train_dataset.labels
    label_to_index = {label: index for index, label in enumerate(labels)}
    train_targets = encode_labels(train_embedding_set.labels, label_to_index)
    val_targets = encode_labels(val_embedding_set.labels, label_to_index)

    head, head_summary = train_classifier_head(
        train_embeddings=train_embedding_set.embeddings,
        train_targets=train_targets,
        val_embeddings=val_embedding_set.embeddings,
        val_targets=val_targets,
        class_count=len(labels),
        config=config,
        device=embedder.device,
    )
    temperature = fit_temperature(head, val_embedding_set.embeddings, val_targets, embedder.device)
    centroids = build_centroids(train_embedding_set.embeddings, train_targets, class_count=len(labels))

    bundle = SpeciesClassifierBundle(
        bioclip_model=config.bioclip_model,
        labels=labels,
        head=head,
        centroids=centroids,
        temperature=temperature,
        thresholds=ThresholdConfig(probability=0.0, margin=0.0, centroid_similarity=0.0),
        device=embedder.device,
        embed_batch_size=config.embed_batch_size,
        top_candidate_count=config.top_candidate_count,
    )

    thresholds, best_threshold_summary, threshold_leaderboard = select_thresholds(
        bundle=bundle,
        val_embeddings=val_embedding_set.embeddings,
        val_targets=val_targets,
        unknown_embeddings=unknown_embeddings,
        config=config,
    )
    bundle.thresholds = thresholds

    calibrated_val_scores = bundle.predict_arrays(val_embedding_set.embeddings)
    final_val_top1 = float(np.mean(calibrated_val_scores["top_indices"] == val_targets)) if len(val_targets) else 0.0

    training_summary = {
        "train_examples": train_dataset.example_count,
        "val_examples": val_dataset.example_count,
        "unknown_examples": len(unknown_paths_text),
        "label_count": len(labels),
        "labels": labels,
        "dropped_train_labels": dropped_train_labels,
        "head_training": head_summary,
        "temperature": round(temperature, 6),
        "threshold_selection_target_precision": config.target_precision,
        "selected_threshold_summary": best_threshold_summary,
        "threshold_leaderboard": threshold_leaderboard,
        "final_val_top1": round(final_val_top1, 6),
    }
    bundle.training_summary = training_summary
    bundle.save(config.output_dir)

    report = {
        "output_dir": str(config.output_dir),
        "classifier_bundle": str(config.output_dir),
        "bioclip_model": config.bioclip_model,
        "device": bundle.device,
        "thresholds": thresholds.to_dict(),
        "training_summary": training_summary,
    }
    (config.output_dir / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report

