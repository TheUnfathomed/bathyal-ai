"""Megalodon detection followed by the production species classifier bundle."""

from __future__ import annotations

import json
import logging
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

from .classifier import SpeciesClassifierBundle, ThresholdConfig
from .data import list_images

if TYPE_CHECKING:
    from .vlm_classification import VlmClassificationClient
    from .vlm_detection import GeminiDetectionClient

try:
    from .vlm_detection import GeminiDetectionClient as _GeminiDetectionClient
    from .vlm_detection import convert_gemini_box
    from .vlm_detection import is_available as _gemini_available
except ImportError:
    _GeminiDetectionClient = None  # type: ignore[assignment,misc]
    convert_gemini_box = None  # type: ignore[assignment]
    _gemini_available = lambda: False  # noqa: E731

try:
    from .vlm_classification import VlmClassificationClient as _VlmClassificationClient
    from .vlm_classification import is_available as _gpt_available
except ImportError:
    _VlmClassificationClient = None  # type: ignore[assignment,misc]
    _gpt_available = lambda: False  # noqa: E731

logger = logging.getLogger(__name__)


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
    min_crop_size: int = 48
    min_detector_confidence: float = 0.3
    vlm_detection_fallback: bool = False
    vlm_classification_fallback: bool = False
    vlm_classification_threshold: float = 0.7
    vlm_max_workers: int = 4


@dataclass
class ImageResult:
    source_path: Path
    source_image: Image.Image
    annotated: Image.Image
    width: int
    height: int
    detections: list[dict[str, object]]
    crops: dict[int, Image.Image]
    image_output: Path
    json_output: Path
    run_dir: Path
    payload: dict[str, object]


def clamp_box(xyxy: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    left, top, right, bottom = [int(round(value)) for value in xyxy]
    left = max(0, min(left, width - 1))
    top = max(0, min(top, height - 1))
    right = max(left + 1, min(right, width))
    bottom = max(top + 1, min(bottom, height))
    return left, top, right, bottom


def choose_color(label: str) -> tuple[int, int, int]:
    if label == "unknown":
        return (214, 39, 40)
    if label.startswith("vlm:"):
        return (31, 119, 180)
    return (44, 160, 44)


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
    source: str = "megalodon",
) -> dict[str, object]:
    predicted_label = str(classification["predicted_label"])
    payload: dict[str, object] = {
        "index": detection_index,
        "source": source,
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


def _write_image_json(image_result: ImageResult) -> None:
    image_result.annotated.save(image_result.image_output)
    image_result.payload["detections"] = image_result.detections
    image_result.json_output.write_text(
        json.dumps(image_result.payload, indent=2), encoding="utf-8"
    )


def _init_vlm_clients(
    config: PipelineConfig,
    classifier_labels: list[str],
) -> tuple[GeminiDetectionClient | None, VlmClassificationClient | None]:
    gemini_client = None
    gpt_client = None

    if config.vlm_detection_fallback:
        if _gemini_available():
            gemini_client = _GeminiDetectionClient()
        else:
            logger.warning("VLM detection fallback requested but unavailable (missing GOOGLE_API_KEY or google-genai package)")

    if config.vlm_classification_fallback:
        if _gpt_available():
            gpt_client = _VlmClassificationClient(known_species=classifier_labels)
        else:
            logger.warning("VLM classification fallback requested but unavailable (missing OPENAI_API_KEY or openai package)")

    return gemini_client, gpt_client


def _run_vlm_postpass(
    image_results: list[ImageResult],
    config: PipelineConfig,
    classifier: SpeciesClassifierBundle,
    active_thresholds: ThresholdConfig,
    candidate_count: int,
    gemini_client: GeminiDetectionClient | None,
    gpt_client: VlmClassificationClient | None,
) -> dict[str, int]:
    vlm_detections_added = 0
    vlm_classifications_added = 0

    executor = ThreadPoolExecutor(max_workers=config.vlm_max_workers)

    # --- Gemini detection fallback for images with zero detections ---
    gemini_futures: dict[int, Future] = {}
    if gemini_client is not None:
        for img_idx, img_result in enumerate(image_results):
            if not img_result.detections:
                gemini_futures[img_idx] = executor.submit(
                    gemini_client.detect, img_result.source_image
                )

    for img_idx, future in gemini_futures.items():
        img_result = image_results[img_idx]
        try:
            gemini_detections = future.result(timeout=120)
        except Exception:
            logger.warning("Gemini detection future failed for %s", img_result.source_path, exc_info=True)
            continue

        for det in gemini_detections:
            box = convert_gemini_box(
                det["box_normalized"], img_result.width, img_result.height
            )
            crop_w = box[2] - box[0]
            crop_h = box[3] - box[1]
            if min(crop_w, crop_h) < config.min_crop_size:
                continue

            crop = img_result.source_image.crop(box)
            classification = classifier.classify(
                crop,
                thresholds=active_thresholds,
                top_candidate_count=candidate_count,
            )

            det_index = len(img_result.detections)
            serialized = serialize_detection(
                detection_index=det_index,
                detector_label=det.get("label", "organism"),
                detector_confidence=1.0,
                box=box,
                classification=classification,
                source="gemini_fallback",
            )
            img_result.detections.append(serialized)
            img_result.crops[det_index] = crop
            vlm_detections_added += 1

            draw_annotation(
                image=img_result.annotated,
                box=box,
                label=str(classification["predicted_label"]),
                probability=float(classification["top_probability"]),
            )

    # --- GPT-5.4 classification fallback for low-confidence detections ---
    classification_futures: dict[tuple[int, int], Future] = {}
    if gpt_client is not None:
        for img_idx, img_result in enumerate(image_results):
            for det in img_result.detections:
                det_index = int(det["index"])
                top_prob = float(det["classification_probability"])
                accepted = bool(det["accepted"])

                if top_prob < config.vlm_classification_threshold or not accepted:
                    crop = img_result.crops.get(det_index)
                    if crop is None:
                        continue

                    box = tuple(det["bbox_xyxy"])
                    candidates = det.get("candidate_rankings", [])

                    classification_futures[(img_idx, det_index)] = executor.submit(
                        gpt_client.classify,
                        crop=crop,
                        frame=img_result.source_image,
                        box=box,
                        bioclip_candidates=candidates,
                    )

    for (img_idx, det_index), future in classification_futures.items():
        img_result = image_results[img_idx]
        try:
            vlm_result = future.result(timeout=120)
        except Exception:
            logger.warning(
                "GPT-5.4 classification future failed for %s detection %d",
                img_result.source_path, det_index, exc_info=True,
            )
            continue

        if vlm_result is None:
            continue

        for det in img_result.detections:
            if int(det["index"]) == det_index:
                det.update(vlm_result)
                if det["status"] == "unknown":
                    det["status"] = "vlm_identified"
                vlm_classifications_added += 1

                box = tuple(det["bbox_xyxy"])
                vlm_label = vlm_result.get("vlm_label", "unknown")
                draw_annotation(
                    image=img_result.annotated,
                    box=box,
                    label=f"vlm:{vlm_label}",
                    probability=0.0,
                )
                break

    executor.shutdown(wait=True)

    # Re-write JSON and annotated images for any images that had VLM updates
    if vlm_detections_added > 0 or vlm_classifications_added > 0:
        for img_result in image_results:
            _write_image_json(img_result)

    return {
        "vlm_detections_added": vlm_detections_added,
        "vlm_classifications_added": vlm_classifications_added,
    }


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
    image_results: list[ImageResult] = []

    # ── Phase 1: Normal pipeline (Megalodon + BioCLIP) ──

    for result in results:
        source_path = Path(result.path)
        source_image = ensure_rgb(result.orig_img)
        annotated = source_image.copy()
        width, height = source_image.size

        detections: list[dict[str, object]] = []
        crops: dict[int, Image.Image] = {}
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

                if detector_confidence < config.min_detector_confidence:
                    continue

                clamped_box = clamp_box(xyxy, width=width, height=height)
                crop_width = clamped_box[2] - clamped_box[0]
                crop_height = clamped_box[3] - clamped_box[1]
                if min(crop_width, crop_height) < config.min_crop_size:
                    continue

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

                serialized = serialize_detection(
                    detection_index=index,
                    detector_label=detector_label,
                    detector_confidence=detector_confidence,
                    box=clamped_box,
                    classification=classification,
                )
                detections.append(serialized)
                crops[index] = crop
                detections_classified += 1

        image_output = run_dir / f"{source_path.stem}.species_classifier.jpg"
        json_output = run_dir / f"{source_path.stem}.species_classifier.json"

        payload: dict[str, object] = {
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
            "vlm_detection_fallback_enabled": config.vlm_detection_fallback,
            "vlm_classification_fallback_enabled": config.vlm_classification_fallback,
            "vlm_classification_threshold": config.vlm_classification_threshold,
            "detections": detections,
        }

        img_result = ImageResult(
            source_path=source_path,
            source_image=source_image,
            annotated=annotated,
            width=width,
            height=height,
            detections=detections,
            crops=crops,
            image_output=image_output,
            json_output=json_output,
            run_dir=run_dir,
            payload=payload,
        )
        _write_image_json(img_result)
        image_results.append(img_result)
        json_outputs.append(str(json_output))

    # ── Phase 2: VLM post-processing pass ──

    gemini_client, gpt_client = _init_vlm_clients(config, classifier.labels)
    vlm_stats: dict[str, int] = {"vlm_detections_added": 0, "vlm_classifications_added": 0}

    if gemini_client is not None or gpt_client is not None:
        vlm_stats = _run_vlm_postpass(
            image_results=image_results,
            config=config,
            classifier=classifier,
            active_thresholds=active_thresholds,
            candidate_count=candidate_count,
            gemini_client=gemini_client,
            gpt_client=gpt_client,
        )

    return {
        "run_dir": str(run_dir),
        "images_processed": len(results),
        "detections_classified": detections_classified,
        "json_outputs": json_outputs,
        **vlm_stats,
    }
