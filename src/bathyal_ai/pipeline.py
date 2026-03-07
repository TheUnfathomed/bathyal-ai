"""Megalodon detection followed by the production species classifier bundle."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

from .classifier import SpeciesClassifierBundle, ThresholdConfig
from .data import list_images


@dataclass(slots=True)
class PipelineConfig:
    source: Path
    classifier_bundle: Path = Path("artifacts/species_classifier")
    detector_model: Path = Path("models/megalodon/mbari-megalodon-yolov8x.pt")
    batch_size: int = 16
    device: str = "auto"
    project: Path | None = None
    name: str | None = None
    exist_ok: bool = False
    save_crops: bool = False
    probability_threshold: float | None = None
    margin_threshold: float | None = None
    centroid_threshold: float | None = None
    top_candidate_count: int | None = None


def clamp_box(xyxy: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    left, top, right, bottom = [int(round(value)) for value in xyxy]
    left = max(0, min(left, width - 1))
    top = max(0, min(top, height - 1))
    right = max(left + 1, min(right, width))
    bottom = max(top + 1, min(bottom, height))
    return left, top, right, bottom


def choose_color(label: str) -> tuple[int, int, int]:
    return (44, 160, 44) if label != "unknown" else (214, 39, 40)


def lookup_detector_label(names: object, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if isinstance(names, list) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def load_annotation_font(image: Image.Image) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    font_size = max(14, int(min(image.size) * 0.025))
    for candidate in ("DejaVuSans.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(candidate, font_size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_annotation(image: Image.Image, box: tuple[int, int, int, int], label: str, probability: float) -> None:
    draw = ImageDraw.Draw(image)
    font = load_annotation_font(image)
    color = choose_color(label)
    left, top, _, _ = box
    draw.rectangle(box, outline=color, width=3)

    text = f"{label} {probability:.2f}"
    text_box = draw.textbbox((left, top), text, font=font)
    text_width = text_box[2] - text_box[0]
    text_height = text_box[3] - text_box[1]
    text_top = max(0, top - text_height - 8)
    text_background = (left, text_top, left + text_width + 10, text_top + text_height + 8)
    draw.rectangle(text_background, fill=color)
    draw.text((left + 5, text_top + 4), text, fill=(255, 255, 255), font=font)


def ensure_rgb(image_array: np.ndarray) -> Image.Image:
    if image_array.ndim == 2:
        return Image.fromarray(image_array).convert("RGB")
    return Image.fromarray(image_array[:, :, ::-1]).convert("RGB")


def resolve_thresholds(bundle: SpeciesClassifierBundle, config: PipelineConfig) -> ThresholdConfig:
    return ThresholdConfig(
        probability=bundle.thresholds.probability if config.probability_threshold is None else config.probability_threshold,
        margin=bundle.thresholds.margin if config.margin_threshold is None else config.margin_threshold,
        centroid_similarity=(
            bundle.thresholds.centroid_similarity if config.centroid_threshold is None else config.centroid_threshold
        ),
    )


def serialize_detection(
    detection_index: int,
    detector_label: str,
    detector_confidence: float,
    box: tuple[int, int, int, int],
    classification: dict[str, object],
) -> dict[str, object]:
    predicted_label = str(classification["predicted_label"])
    payload = {
        "index": detection_index,
        "bbox_xyxy": list(box),
        "detector_label": detector_label,
        "detector_confidence": round(detector_confidence, 6),
        "classification_label": predicted_label,
        "classification_probability": float(classification["top_probability"]),
        "classification_margin": float(classification["margin"]),
        "classification_centroid_similarity": float(classification["centroid_similarity"]),
        "top_candidate_label": str(classification["top_label"]),
        "second_best_label": classification["second_best_label"],
        "second_best_probability": float(classification["second_best_probability"]),
        "accepted": bool(classification["accepted"]),
        "rejection_reasons": classification["rejection_reasons"],
        "status": "known" if predicted_label != "unknown" else "unknown",
        "candidate_rankings": classification["candidate_rankings"],
        "thresholds": classification["thresholds"],
    }
    if "crop_path" in classification:
        payload["crop_path"] = classification["crop_path"]
    return payload


def run_pipeline(config: PipelineConfig) -> dict[str, object]:
    if not config.detector_model.exists():
        raise FileNotFoundError(f"Detector model not found: {config.detector_model}")
    if not config.classifier_bundle.exists():
        raise FileNotFoundError(f"Classifier bundle not found: {config.classifier_bundle}")

    source_images = list_images(config.source)
    if not source_images:
        raise ValueError(f"No source images found under {config.source}")

    classifier = SpeciesClassifierBundle.load(
        config.classifier_bundle,
        device=config.device,
        embed_batch_size=config.batch_size,
    )
    active_thresholds = resolve_thresholds(classifier, config)
    candidate_count = classifier.top_candidate_count if config.top_candidate_count is None else config.top_candidate_count

    detector = YOLO(str(config.detector_model))
    predict_kwargs: dict[str, object] = {
        "source": str(config.source),
        "save": True,
        "exist_ok": config.exist_ok,
        "verbose": False,
    }
    if config.project is not None:
        predict_kwargs["project"] = str(config.project)
    if config.name is not None:
        predict_kwargs["name"] = config.name
    results = detector.predict(**predict_kwargs)

    if not results:
        return {
            "run_dir": "",
            "images_processed": 0,
            "detections_classified": 0,
            "json_outputs": [],
        }

    run_dir = Path(results[0].save_dir)
    json_outputs: list[str] = []
    detections_classified = 0

    for result in results:
        source_path = Path(result.path)
        source_image = ensure_rgb(result.orig_img)
        annotated = source_image.copy()
        width, height = source_image.size

        detections: list[dict[str, object]] = []
        crop_dir = run_dir / f"{source_path.stem}_crops"
        if config.save_crops:
            crop_dir.mkdir(parents=True, exist_ok=True)

        boxes = result.boxes
        if boxes is not None:
            for index, box in enumerate(boxes):
                xyxy = [float(value) for value in box.xyxy[0].tolist()]
                detector_confidence = float(box.conf.item())
                class_id = int(box.cls.item()) if box.cls is not None else -1
                detector_label = lookup_detector_label(result.names, class_id)

                clamped_box = clamp_box(xyxy, width=width, height=height)
                crop = source_image.crop(clamped_box)
                classification = classifier.classify(
                    crop,
                    thresholds=active_thresholds,
                    top_candidate_count=candidate_count,
                )

                if config.save_crops:
                    crop_path = crop_dir / f"{source_path.stem}_crop_{index:03d}.jpg"
                    crop.save(crop_path)
                    classification["crop_path"] = str(crop_path)

                draw_annotation(
                    image=annotated,
                    box=clamped_box,
                    label=str(classification["predicted_label"]),
                    probability=float(classification["top_probability"]),
                )
                detections.append(
                    serialize_detection(
                        detection_index=index,
                        detector_label=detector_label,
                        detector_confidence=detector_confidence,
                        box=clamped_box,
                        classification=classification,
                    )
                )
                detections_classified += 1

        image_output = run_dir / f"{source_path.stem}.species_classifier.jpg"
        json_output = run_dir / f"{source_path.stem}.species_classifier.json"
        annotated.save(image_output)

        payload = {
            "source_image": str(source_path),
            "megalodon_output_image": str(run_dir / source_path.name),
            "classified_output_image": str(image_output),
            "classifier_bundle": str(config.classifier_bundle),
            "classifier_labels": classifier.labels,
            "classifier_label_count": len(classifier.labels),
            "classifier_strategy": "bioclip_embedding_linear_head_with_open_set_gating",
            "bioclip_model": classifier.bioclip_model,
            "bundle_created_at": classifier.created_at,
            "bundle_thresholds": classifier.thresholds.to_dict(),
            "active_thresholds": active_thresholds.to_dict(),
            "top_candidate_count": candidate_count,
            "detections": detections,
        }
        json_output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        json_outputs.append(str(json_output))

    return {
        "run_dir": str(run_dir),
        "images_processed": len(results),
        "detections_classified": detections_classified,
        "json_outputs": json_outputs,
    }
