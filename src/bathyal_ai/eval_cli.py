"""CLI entry point for evaluation sweeps."""

from __future__ import annotations

import argparse
from pathlib import Path

from .evaluation import EvaluationConfig, run_evaluation



def parse_float_list(raw: str) -> list[float]:
    values = [value.strip() for value in raw.split(",") if value.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated float")
    return [float(value) for value in values]



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Bathyal AI classifier bundle over labeled crops.",
    )
    parser.add_argument("--source", required=True, help="Labeled image file or directory to evaluate.")
    parser.add_argument(
        "--classifier-bundle",
        required=True,
        help="Path to the classifier bundle produced by bathyal-ai-train.",
    )
    parser.add_argument(
        "--unknown-dir",
        default=None,
        help="Optional directory of out-of-taxonomy crops that should be rejected as unknown.",
    )
    parser.add_argument(
        "--probability-thresholds",
        type=parse_float_list,
        default=None,
        help="Optional comma-separated calibrated probability thresholds to sweep.",
    )
    parser.add_argument(
        "--margin-thresholds",
        type=parse_float_list,
        default=None,
        help="Optional comma-separated probability margin thresholds to sweep.",
    )
    parser.add_argument(
        "--centroid-thresholds",
        type=parse_float_list,
        default=None,
        help="Optional comma-separated centroid similarity thresholds to sweep.",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for BioCLIP 2 embedding.")
    parser.add_argument("--device", default="auto", help="Execution device: auto, cpu, or cuda.")
    parser.add_argument(
        "--cache-dir",
        default=".cache/embeddings",
        help="Optional embedding cache directory.",
    )
    parser.add_argument("--output-json", default=None, help="Optional path to write the full evaluation JSON.")
    return parser



def main() -> None:
    args = build_parser().parse_args()
    config = EvaluationConfig(
        source=Path(args.source),
        classifier_bundle=Path(args.classifier_bundle),
        unknown_dir=Path(args.unknown_dir) if args.unknown_dir else None,
        device=args.device,
        batch_size=args.batch_size,
        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
        probability_thresholds=args.probability_thresholds,
        margin_thresholds=args.margin_thresholds,
        centroid_thresholds=args.centroid_thresholds,
        output_json=Path(args.output_json) if args.output_json else None,
    )
    result = run_evaluation(config)
    print(f"Classifier labels: {result['label_count']}")
    print(f"Known examples: {result['known_example_count']}")
    print(f"Unknown examples: {result['unknown_example_count']}")
    print("Threshold sweep:")
    for summary in result["summaries"]:
        precision = summary["precision_when_accepted"]
        precision_text = "n/a" if precision is None else f"{precision:.3f}"
        print(
            "  "
            f"prob={summary['probability_threshold']:.2f} "
            f"margin={summary['margin_threshold']:.2f} "
            f"centroid={summary['centroid_threshold']:.2f} "
            f"coverage={summary['known_coverage']:.3f} "
            f"precision={precision_text} "
            f"known_fallback={summary['known_fallback_rate']:.3f} "
            f"unknown_accept={summary['unknown_accept_rate']:.3f} "
            f"top1={summary['top1_known_accuracy']:.3f}"
        )


if __name__ == "__main__":
    main()
