"""Shared open-set metrics for species classification."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .classifier import ThresholdConfig


def apply_thresholds(score_arrays: dict[str, np.ndarray], thresholds: ThresholdConfig) -> np.ndarray:
    return (
        (score_arrays["top_probabilities"] >= thresholds.probability)
        & (score_arrays["margins"] >= thresholds.margin)
        & (score_arrays["predicted_centroid_similarities"] >= thresholds.centroid_similarity)
    )


def summarize_thresholds(
    known_score_arrays: dict[str, np.ndarray],
    known_targets: np.ndarray,
    thresholds: ThresholdConfig,
    unknown_score_arrays: dict[str, np.ndarray] | None = None,
) -> dict[str, object]:
    known_accepted_mask = apply_thresholds(known_score_arrays, thresholds)
    known_total = int(len(known_targets))
    known_top_indices = known_score_arrays["top_indices"]

    correct_accepted = int(np.sum((known_top_indices == known_targets) & known_accepted_mask))
    wrong_accepted = int(np.sum((known_top_indices != known_targets) & known_accepted_mask))
    known_accepted = int(np.sum(known_accepted_mask))
    known_fallback = known_total - known_accepted

    unknown_total = 0
    unknown_accepted = 0
    if unknown_score_arrays is not None and len(unknown_score_arrays["top_indices"]) > 0:
        unknown_accepted_mask = apply_thresholds(unknown_score_arrays, thresholds)
        unknown_total = int(len(unknown_score_arrays["top_indices"]))
        unknown_accepted = int(np.sum(unknown_accepted_mask))

    total_accepts = known_accepted + unknown_accepted
    precision = correct_accepted / total_accepts if total_accepts else None
    known_top1_accuracy = float(np.mean(known_top_indices == known_targets)) if known_total else 0.0

    return {
        "probability_threshold": thresholds.probability,
        "margin_threshold": thresholds.margin,
        "centroid_threshold": thresholds.centroid_similarity,
        "known_total": known_total,
        "unknown_total": unknown_total,
        "accepted_known": known_accepted,
        "fallback_known": known_fallback,
        "correct_accepted": correct_accepted,
        "wrong_accepted_known": wrong_accepted,
        "unknown_accepted": unknown_accepted,
        "known_accept_rate": round(known_accepted / known_total, 6) if known_total else 0.0,
        "known_coverage": round(correct_accepted / known_total, 6) if known_total else 0.0,
        "known_fallback_rate": round(known_fallback / known_total, 6) if known_total else 0.0,
        "wrong_accept_rate_known": round(wrong_accepted / known_total, 6) if known_total else 0.0,
        "unknown_accept_rate": round(unknown_accepted / unknown_total, 6) if unknown_total else 0.0,
        "precision_when_accepted": round(precision, 6) if precision is not None else None,
        "top1_known_accuracy": round(known_top1_accuracy, 6),
    }


def rejection_reasons_for_index(
    score_arrays: dict[str, np.ndarray],
    thresholds: ThresholdConfig,
    index: int,
) -> list[str]:
    reasons: list[str] = []
    if float(score_arrays["top_probabilities"][index]) < thresholds.probability:
        reasons.append("probability")
    if float(score_arrays["margins"][index]) < thresholds.margin:
        reasons.append("margin")
    if float(score_arrays["predicted_centroid_similarities"][index]) < thresholds.centroid_similarity:
        reasons.append("centroid_similarity")
    return reasons


def build_known_prediction_records(
    paths: Sequence[str],
    true_labels: Sequence[str],
    score_arrays: dict[str, np.ndarray],
    thresholds: ThresholdConfig,
    labels: Sequence[str],
) -> list[dict[str, object]]:
    accepted_mask = apply_thresholds(score_arrays, thresholds)
    records: list[dict[str, object]] = []
    for index, path in enumerate(paths):
        top_index = int(score_arrays["top_indices"][index])
        second_index = int(score_arrays["second_indices"][index])
        top_label = labels[top_index]
        second_label = labels[second_index] if second_index >= 0 else None
        accepted = bool(accepted_mask[index])
        records.append(
            {
                "path": path,
                "true_label": str(true_labels[index]),
                "predicted_label": top_label if accepted else "unknown",
                "top_label": top_label,
                "top_probability": round(float(score_arrays["top_probabilities"][index]), 6),
                "second_best_label": second_label,
                "second_best_probability": round(float(score_arrays["second_probabilities"][index]), 6),
                "margin": round(float(score_arrays["margins"][index]), 6),
                "centroid_similarity": round(float(score_arrays["predicted_centroid_similarities"][index]), 6),
                "accepted": accepted,
                "rejection_reasons": rejection_reasons_for_index(score_arrays, thresholds, index) if not accepted else [],
            }
        )
    return records


def BuildPerClassMetrics(
    predictions: list[dict[str, object]],
    labels: Sequence[str],
) -> dict[str, object]:
    confusion: dict[str, dict[str, int]] = {label: {} for label in labels}
    confusion["unknown"] = {}
    per_class: dict[str, dict[str, int]] = {
        label: {"tp": 0, "fp": 0, "fn": 0, "support": 0, "accepted": 0, "rejected": 0}
        for label in labels
    }

    for pred in predictions:
        true_label = str(pred["true_label"])
        predicted_label = str(pred["predicted_label"])
        accepted = bool(pred["accepted"])

        if true_label in per_class:
            per_class[true_label]["support"] += 1
            if accepted:
                per_class[true_label]["accepted"] += 1
            else:
                per_class[true_label]["rejected"] += 1

        row = confusion.get(true_label, confusion.setdefault(true_label, {}))
        row[predicted_label] = row.get(predicted_label, 0) + 1

        if predicted_label != "unknown":
            if predicted_label == true_label:
                per_class[predicted_label]["tp"] += 1
            else:
                per_class[predicted_label]["fp"] += 1
                if true_label in per_class:
                    per_class[true_label]["fn"] += 1
        else:
            if true_label in per_class:
                per_class[true_label]["fn"] += 1

    class_metrics: dict[str, dict[str, object]] = {}
    for label in labels:
        stats = per_class[label]
        tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else None
        recall = tp / (tp + fn) if (tp + fn) > 0 else None
        f1 = (2 * precision * recall / (precision + recall)) if precision and recall and (precision + recall) > 0 else None
        class_metrics[label] = {
            "precision": round(precision, 4) if precision is not None else None,
            "recall": round(recall, 4) if recall is not None else None,
            "f1": round(f1, 4) if f1 is not None else None,
            "support": stats["support"],
            "accepted": stats["accepted"],
            "rejected": stats["rejected"],
        }

    return {
        "per_class": class_metrics,
        "confusion_matrix": {k: dict(v) for k, v in confusion.items() if v},
    }


def build_unknown_prediction_records(
    paths: Sequence[str],
    score_arrays: dict[str, np.ndarray],
    thresholds: ThresholdConfig,
    labels: Sequence[str],
) -> list[dict[str, object]]:
    accepted_mask = apply_thresholds(score_arrays, thresholds)
    records: list[dict[str, object]] = []
    for index, path in enumerate(paths):
        top_index = int(score_arrays["top_indices"][index])
        second_index = int(score_arrays["second_indices"][index])
        top_label = labels[top_index]
        second_label = labels[second_index] if second_index >= 0 else None
        accepted = bool(accepted_mask[index])
        records.append(
            {
                "path": path,
                "true_label": "unknown",
                "predicted_label": top_label if accepted else "unknown",
                "top_label": top_label,
                "top_probability": round(float(score_arrays["top_probabilities"][index]), 6),
                "second_best_label": second_label,
                "second_best_probability": round(float(score_arrays["second_probabilities"][index]), 6),
                "margin": round(float(score_arrays["margins"][index]), 6),
                "centroid_similarity": round(float(score_arrays["predicted_centroid_similarities"][index]), 6),
                "accepted": accepted,
                "rejection_reasons": rejection_reasons_for_index(score_arrays, thresholds, index) if not accepted else [],
            }
        )
    return records
