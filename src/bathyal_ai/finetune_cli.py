"""CLI entry point for LoRA fine-tuning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .finetune import FinetuneConfig, run_finetune


def parse_float_list(raw: str) -> list[float]:
    values = [value.strip() for value in raw.split(",") if value.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated float")
    return [float(value) for value in values]


def parse_int_list(raw: str) -> list[int]:
    values = [value.strip() for value in raw.split(",") if value.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated int")
    return [int(value) for value in values]


def parse_str_list(raw: str) -> list[str]:
    return [value.strip() for value in raw.split(",") if value.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="LoRA fine-tune BioCLIP 2 for species classification.",
    )
    parser.add_argument("--train-dir", required=True, help="Directory of training crops grouped by species.")
    parser.add_argument("--val-dir", required=True, help="Directory of validation crops grouped by species.")
    parser.add_argument("--unknown-dir", default=None, help="Optional directory of unknown crops for threshold calibration.")
    parser.add_argument("--output-dir", default="artifacts/species_classifier_lora", help="Output directory for the fine-tuned bundle.")
    parser.add_argument("--bioclip-model", default="hf-hub:imageomics/bioclip-2", help="BioCLIP model identifier.")
    parser.add_argument("--resume-from", default=None, help="Path to a prior output directory for warm-start initialization.")
    parser.add_argument("--resume-checkpoint", default=None, help="Path to a checkpoint file to resume an interrupted training run.")
    parser.add_argument("--checkpoint-every", type=int, default=None, help="Save a checkpoint every N epochs (also saves on best-val improvement).")
    parser.add_argument("--lora-rank", type=int, default=None, help="LoRA rank (default: 8, or inherited from --resume-from).")
    parser.add_argument("--lora-alpha", type=float, default=None, help="LoRA alpha scaling factor (default: 16.0, or inherited from --resume-from).")
    parser.add_argument("--lora-target-blocks", type=parse_int_list, default=None, help="Comma-separated block indices for LoRA (default: all).")
    parser.add_argument("--lora-targets", type=parse_str_list, default=None, help="Comma-separated projections to adapt: q,k,v,o,mlp_fc,mlp_proj.")
    parser.add_argument("--lora-dropout", type=float, default=None, help="LoRA dropout rate (default: 0.0, or inherited from --resume-from).")
    parser.add_argument("--lora-lr", type=float, default=1e-4, help="Learning rate for LoRA parameters.")
    parser.add_argument("--head-lr", type=float, default=5e-3, help="Learning rate for classifier head.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay.")
    parser.add_argument("--label-smoothing", type=float, default=0.05, help="Label smoothing.")
    parser.add_argument("--epochs", type=int, default=20, help="Maximum training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--warmup-epochs", type=int, default=2, help="Linear warmup epochs.")
    parser.add_argument("--min-examples-per-label", type=int, default=2, help="Drop labels with fewer examples.")
    parser.add_argument("--target-precision", type=float, default=0.98, help="Target precision for threshold selection.")
    parser.add_argument("--early-stopping-patience", type=int, default=5, help="Early stopping patience in epochs.")
    parser.add_argument("--mixed-precision", action="store_true", default=True, help="Enable mixed precision (bf16/fp16).")
    parser.add_argument("--no-mixed-precision", action="store_true", help="Disable mixed precision.")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker count.")
    parser.add_argument("--device", default="auto", help="Device: auto, cpu, or cuda.")
    parser.add_argument("--seed", type=int, default=13, help="Random seed.")
    parser.add_argument(
        "--probability-thresholds", type=parse_float_list,
        default=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
        help="Probability thresholds to sweep.",
    )
    parser.add_argument(
        "--margin-thresholds", type=parse_float_list,
        default=[0.0, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.25],
        help="Margin thresholds to sweep.",
    )
    parser.add_argument(
        "--centroid-thresholds", type=parse_float_list,
        default=[0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        help="Centroid similarity thresholds to sweep.",
    )
    return parser


def _resolve_lora_defaults(args: argparse.Namespace) -> None:
    prior_lora: dict[str, object] = {}
    if args.resume_from:
        meta_path = Path(args.resume_from) / "metadata.json"
        if meta_path.exists():
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            prior_lora = metadata.get("lora_config", {})

    if args.lora_rank is None:
        args.lora_rank = prior_lora.get("rank", 8)
    if args.lora_alpha is None:
        args.lora_alpha = prior_lora.get("alpha", 16.0)
    if args.lora_target_blocks is None:
        args.lora_target_blocks = prior_lora.get("target_blocks")
    if args.lora_targets is None:
        prior_targets = prior_lora.get("targets")
        args.lora_targets = prior_targets if prior_targets is not None else ["q", "k", "v", "o"]
    if args.lora_dropout is None:
        args.lora_dropout = float(prior_lora.get("dropout", 0.0))


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.resume_from and args.resume_checkpoint:
        parser.error("--resume-from and --resume-checkpoint are mutually exclusive")

    _resolve_lora_defaults(args)

    lora_targets = args.lora_targets if isinstance(args.lora_targets, list) else parse_str_list(args.lora_targets)

    config = FinetuneConfig(
        train_dir=Path(args.train_dir),
        val_dir=Path(args.val_dir),
        output_dir=Path(args.output_dir),
        unknown_dir=Path(args.unknown_dir) if args.unknown_dir else None,
        bioclip_model=args.bioclip_model,
        resume_from=Path(args.resume_from) if args.resume_from else None,
        resume_checkpoint=Path(args.resume_checkpoint) if args.resume_checkpoint else None,
        checkpoint_every=args.checkpoint_every,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_target_blocks=args.lora_target_blocks,
        lora_targets=lora_targets,
        lora_dropout=args.lora_dropout,
        lora_lr=args.lora_lr,
        head_lr=args.head_lr,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        epochs=args.epochs,
        batch_size=args.batch_size,
        warmup_epochs=args.warmup_epochs,
        min_examples_per_label=args.min_examples_per_label,
        target_precision=args.target_precision,
        early_stopping_patience=args.early_stopping_patience,
        mixed_precision=not args.no_mixed_precision,
        num_workers=args.num_workers,
        device=args.device,
        seed=args.seed,
        probability_thresholds=args.probability_thresholds,
        margin_thresholds=args.margin_thresholds,
        centroid_thresholds=args.centroid_thresholds,
    )

    result = run_finetune(config)
    print(f"\nClassifier bundle: {result['classifier_bundle']}")
    print(f"Device: {result['device']}")
    print("Selected thresholds:")
    print(f"  probability={result['thresholds']['probability']:.3f}")
    print(f"  margin={result['thresholds']['margin']:.3f}")
    print(f"  centroid_similarity={result['thresholds']['centroid_similarity']:.3f}")
    summary = result["training_summary"]
    print(f"Labels: {summary['label_count']}")
    print(f"Validation top1: {summary['final_val_top1']:.3f}")


if __name__ == "__main__":
    main()
