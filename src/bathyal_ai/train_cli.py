"""CLI entry point for classifier training."""

from __future__ import annotations

import argparse
from pathlib import Path

from .training import TrainingConfig, train_classifier



def parse_float_list(raw: str) -> list[float]:
    values = [value.strip() for value in raw.split(",") if value.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated float")
    return [float(value) for value in values]



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the production Bathyal AI species classifier bundle.",
    )
    parser.add_argument("--train-dir", required=True, help="Directory of training crops grouped by species.")
    parser.add_argument("--val-dir", required=True, help="Directory of validation crops grouped by species.")
    parser.add_argument(
        "--unknown-dir",
        default=None,
        help="Optional directory of unknown or out-of-taxonomy crops for threshold calibration.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/species_classifier",
        help="Directory where the trained classifier bundle will be written.",
    )
    parser.add_argument(
        "--bioclip-model",
        default="hf-hub:imageomics/bioclip-2",
        help="BioCLIP 2 model identifier understood by open_clip.",
    )
    parser.add_argument(
        "--probability-thresholds",
        type=parse_float_list,
        default=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
        help="Comma-separated calibrated probability thresholds to sweep.",
    )
    parser.add_argument(
        "--margin-thresholds",
        type=parse_float_list,
        default=[0.0, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.25],
        help="Comma-separated probability margin thresholds to sweep.",
    )
    parser.add_argument(
        "--centroid-thresholds",
        type=parse_float_list,
        default=[0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        help="Comma-separated centroid similarity thresholds to sweep.",
    )
    parser.add_argument(
        "--target-precision",
        type=float,
        default=0.98,
        help="Minimum accepted-label precision target during threshold calibration.",
    )
    parser.add_argument(
        "--min-examples-per-label",
        type=int,
        default=2,
        help="Drop training labels with fewer than this many examples.",
    )
    parser.add_argument("--epochs", type=int, default=200, help="Maximum classifier-head training epochs.")
    parser.add_argument(
        "--classifier-batch-size",
        type=int,
        default=64,
        help="Batch size for training the classifier head over BioCLIP embeddings.",
    )
    parser.add_argument("--learning-rate", type=float, default=5e-3, help="Classifier-head learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Classifier-head weight decay.")
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.05,
        help="Cross-entropy label smoothing for the classifier head.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=30,
        help="Stop training if validation loss stops improving for this many epochs.",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for BioCLIP 2 embedding.")
    parser.add_argument("--device", default="auto", help="Execution device: auto, cpu, or cuda.")
    parser.add_argument(
        "--cache-dir",
        default=".cache/embeddings",
        help="Optional embedding cache directory.",
    )
    parser.add_argument(
        "--top-candidates",
        type=int,
        default=5,
        help="How many candidate species to retain in classifier diagnostics.",
    )
    parser.add_argument("--seed", type=int, default=13, help="Random seed.")
    return parser



def main() -> None:
    args = build_parser().parse_args()
    config = TrainingConfig(
        train_dir=Path(args.train_dir),
        val_dir=Path(args.val_dir),
        output_dir=Path(args.output_dir),
        unknown_dir=Path(args.unknown_dir) if args.unknown_dir else None,
        bioclip_model=args.bioclip_model,
        embed_batch_size=args.batch_size,
        classifier_batch_size=args.classifier_batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        min_examples_per_label=args.min_examples_per_label,
        target_precision=args.target_precision,
        probability_thresholds=args.probability_thresholds,
        margin_thresholds=args.margin_thresholds,
        centroid_thresholds=args.centroid_thresholds,
        device=args.device,
        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
        top_candidate_count=args.top_candidates,
        seed=args.seed,
        early_stopping_patience=args.early_stopping_patience,
    )
    result = train_classifier(config)
    print(f"Classifier bundle: {result['classifier_bundle']}")
    print(f"Device: {result['device']}")
    print("Selected thresholds:")
    print(f"  probability={result['thresholds']['probability']:.3f}")
    print(f"  margin={result['thresholds']['margin']:.3f}")
    print(f"  centroid_similarity={result['thresholds']['centroid_similarity']:.3f}")
    summary = result["training_summary"]
    print(f"Train examples: {summary['train_examples']}")
    print(f"Validation examples: {summary['val_examples']}")
    print(f"Unknown calibration examples: {summary['unknown_examples']}")
    print(f"Labels: {summary['label_count']}")
    print(f"Validation top1: {summary['final_val_top1']:.3f}")


if __name__ == "__main__":
    main()
