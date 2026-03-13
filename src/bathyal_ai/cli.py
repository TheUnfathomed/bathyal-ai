"""Command line entry point for Bathyal AI inference."""

from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import PipelineConfig, run_pipeline



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Megalodon detection and the trained Bathyal AI species classifier bundle.",
    )
    parser.add_argument("--source", required=True, help="Image file or directory to process.")
    parser.add_argument(
        "--classifier-bundle",
        default="artifacts/species_classifier",
        help="Path to a trained classifier bundle produced by bathyal-ai-train.",
    )
    parser.add_argument(
        "--detector-model",
        default="models/megalodon/mbari-megalodon-yolov8x.pt",
        help="Path to the Megalodon YOLO weights.",
    )
    parser.add_argument(
        "--probability-threshold",
        type=float,
        default=None,
        help="Optional override for the minimum calibrated species probability.",
    )
    parser.add_argument(
        "--margin-threshold",
        type=float,
        default=None,
        help="Optional override for the minimum gap between the best and second-best species probabilities.",
    )
    parser.add_argument(
        "--centroid-threshold",
        type=float,
        default=None,
        help="Optional override for the minimum similarity to the predicted class centroid.",
    )
    parser.add_argument(
        "--top-candidates",
        type=int,
        default=None,
        help="Optional override for how many candidate species to keep in output diagnostics.",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for BioCLIP 2 embedding.")
    parser.add_argument("--device", default="auto", help="Execution device: auto, cpu, or cuda.")
    parser.add_argument("--project", default=None, help="Optional Ultralytics project directory override.")
    parser.add_argument("--name", default=None, help="Optional Ultralytics run name override.")
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="Reuse the run directory instead of auto-incrementing it.",
    )
    parser.add_argument(
        "--save-crops",
        action="store_true",
        help="Save each detection crop used for classification.",
    )
    parser.add_argument(
        "--min-crop-size",
        type=int,
        default=48,
        help="Skip detections smaller than this on either side (pixels).",
    )
    parser.add_argument(
        "--min-detector-confidence",
        type=float,
        default=0.3,
        help="Skip detections below this detector confidence.",
    )
    parser.add_argument(
        "--vlm-detection-fallback",
        action="store_true",
        help="Use Gemini 3 Flash to detect organisms when Megalodon finds nothing. Requires GOOGLE_API_KEY.",
    )
    parser.add_argument(
        "--vlm-classification-fallback",
        action="store_true",
        help="Use GPT-5.4 to identify species when BioCLIP confidence is low. Requires OPENAI_API_KEY.",
    )
    parser.add_argument(
        "--vlm-classification-threshold",
        type=float,
        default=0.7,
        help="BioCLIP top probability below which GPT-5.4 classification fallback triggers.",
    )
    parser.add_argument(
        "--vlm-max-workers",
        type=int,
        default=4,
        help="Thread pool size for concurrent VLM API calls.",
    )
    return parser



def main() -> None:
    args = build_parser().parse_args()
    config = PipelineConfig(
        source=Path(args.source),
        classifier_bundle=Path(args.classifier_bundle),
        detector_model=Path(args.detector_model),
        batch_size=args.batch_size,
        device=args.device,
        project=Path(args.project) if args.project else None,
        name=args.name,
        exist_ok=args.exist_ok,
        save_crops=args.save_crops,
        probability_threshold=args.probability_threshold,
        margin_threshold=args.margin_threshold,
        centroid_threshold=args.centroid_threshold,
        top_candidate_count=args.top_candidates,
        min_crop_size=args.min_crop_size,
        min_detector_confidence=args.min_detector_confidence,
        vlm_detection_fallback=args.vlm_detection_fallback,
        vlm_classification_fallback=args.vlm_classification_fallback,
        vlm_classification_threshold=args.vlm_classification_threshold,
        vlm_max_workers=args.vlm_max_workers,
    )
    summary = run_pipeline(config)
    print(f"Run directory: {summary['run_dir']}")
    print(f"Images processed: {summary['images_processed']}")
    print(f"Detections classified: {summary['detections_classified']}")
    if summary.get("vlm_detections_added"):
        print(f"VLM detection fallbacks: {summary['vlm_detections_added']}")
    if summary.get("vlm_classifications_added"):
        print(f"VLM classification fallbacks: {summary['vlm_classifications_added']}")
    if summary["json_outputs"]:
        print("Classification outputs:")
        for output in summary["json_outputs"]:
            print(f"  - {output}")


if __name__ == "__main__":
    main()
