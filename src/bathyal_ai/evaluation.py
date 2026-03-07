"""Evaluation helpers for the production Bathyal AI classifier bundle."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .classifier import SpeciesClassifierBundle, ThresholdConfig
from .data import ensure_labels_subset, index_labeled_dataset, list_images
from .metrics import BuildPerClassMetrics, build_known_prediction_records, build_unknown_prediction_records, summarize_thresholds


@dataclass(slots=True)
class EvaluationConfig:
    source: Path
    classifier_bundle: Path
    unknown_dir: Path | None = None
    device: str = "auto"
    batch_size: int = 16
    cache_dir: Path | None = Path(".cache/embeddings")
    probability_thresholds: list[float] | None = None
    margin_thresholds: list[float] | None = None
    centroid_thresholds: list[float] | None = None
    output_json: Path | None = None



def run_evaluation(config: EvaluationConfig) -> dict[str, object]:
    classifier = SpeciesClassifierBundle.load(
        config.classifier_bundle,
        device=config.device,
        embed_batch_size=config.batch_size,
    )

    known_dataset = index_labeled_dataset(config.source)
    ensure_labels_subset(known_dataset, classifier.labels, "Evaluation dataset")

    known_targets = [example.label for example in known_dataset.examples]
    known_target_indices = np.asarray([classifier.label_to_index[label] for label in known_targets], dtype=np.int64)

    known_embedding_set = classifier.embedder.embed_examples(
        known_dataset.examples,
        cache_dir=config.cache_dir,
        cache_namespace="evaluation-known",
    )
    known_score_arrays = classifier.predict_arrays(known_embedding_set.embeddings)

    unknown_paths = list_images(config.unknown_dir) if config.unknown_dir is not None else []
    unknown_embedding_set = None
    unknown_score_arrays = None
    if unknown_paths:
        unknown_embedding_set = classifier.embedder.embed_unlabeled_paths(
            unknown_paths,
            cache_dir=config.cache_dir,
            cache_namespace="evaluation-unknown",
        )
        unknown_score_arrays = classifier.predict_arrays(unknown_embedding_set.embeddings)

    probability_thresholds = config.probability_thresholds or [classifier.thresholds.probability]
    margin_thresholds = config.margin_thresholds or [classifier.thresholds.margin]
    centroid_thresholds = config.centroid_thresholds or [classifier.thresholds.centroid_similarity]

    summaries: list[dict[str, object]] = []
    for probability_threshold in probability_thresholds:
        for margin_threshold in margin_thresholds:
            for centroid_threshold in centroid_thresholds:
                thresholds = ThresholdConfig(
                    probability=probability_threshold,
                    margin=margin_threshold,
                    centroid_similarity=centroid_threshold,
                )
                summaries.append(
                    summarize_thresholds(
                        known_score_arrays=known_score_arrays,
                        known_targets=known_target_indices,
                        thresholds=thresholds,
                        unknown_score_arrays=unknown_score_arrays,
                    )
                )

    default_thresholds = ThresholdConfig(
        probability=classifier.thresholds.probability,
        margin=classifier.thresholds.margin,
        centroid_similarity=classifier.thresholds.centroid_similarity,
    )
    default_summary = summarize_thresholds(
        known_score_arrays=known_score_arrays,
        known_targets=known_target_indices,
        thresholds=default_thresholds,
        unknown_score_arrays=unknown_score_arrays,
    )

    known_predictions = build_known_prediction_records(
        paths=known_embedding_set.paths,
        true_labels=known_targets,
        score_arrays=known_score_arrays,
        thresholds=default_thresholds,
        labels=classifier.labels,
    )
    unknown_predictions = (
        build_unknown_prediction_records(
            paths=unknown_embedding_set.paths,
            score_arrays=unknown_score_arrays,
            thresholds=default_thresholds,
            labels=classifier.labels,
        )
        if unknown_embedding_set is not None and unknown_score_arrays is not None
        else []
    )

    result = {
        "source": str(config.source),
        "classifier_bundle": str(config.classifier_bundle),
        "bioclip_model": classifier.bioclip_model,
        "classifier_labels": classifier.labels,
        "label_count": len(classifier.labels),
        "known_example_count": known_dataset.example_count,
        "unknown_example_count": len(unknown_paths),
        "bundle_thresholds": classifier.thresholds.to_dict(),
        "default_threshold_summary": default_summary,
        "summaries": summaries,
        "known_predictions": known_predictions,
        "unknown_predictions": unknown_predictions,
        "per_class_metrics": BuildPerClassMetrics(known_predictions, classifier.labels),
    }

    if config.output_json is not None:
        config.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    return result
