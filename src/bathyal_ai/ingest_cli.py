"""CLI entry point for dataset ingestion."""

from __future__ import annotations

import argparse
from pathlib import Path

from .ingest import FathomNetIngestConfig, ingest_fathomnet



def parse_csv(raw: str) -> list[str]:
    return [value.strip() for value in raw.split(",") if value.strip()]



def load_concepts(args: argparse.Namespace) -> list[str]:
    concepts: list[str] = []
    for concept in args.concept or []:
        concepts.extend(parse_csv(concept))
    if args.concept_file:
        lines = Path(args.concept_file).read_text(encoding="utf-8").splitlines()
        concepts.extend([line.strip() for line in lines if line.strip() and not line.strip().startswith("#")])
    deduped: list[str] = []
    seen: set[str] = set()
    for concept in concepts:
        if concept not in seen:
            deduped.append(concept)
            seen.add(concept)
    return deduped



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingest labeled crop data for Bathyal AI.")
    subparsers = parser.add_subparsers(dest="source", required=True)

    fathomnet_parser = subparsers.add_parser(
        "fathomnet",
        help="Download FathomNet images and turn matching annotated boxes into classifier crops.",
    )
    fathomnet_parser.add_argument(
        "--concept",
        action="append",
        default=[],
        help="Concept to ingest. Repeat or pass comma-separated values.",
    )
    fathomnet_parser.add_argument(
        "--concept-file",
        default=None,
        help="Optional newline-delimited concept file.",
    )
    fathomnet_parser.add_argument(
        "--output-dir",
        default="datasets/fathomnet_seed",
        help="Dataset root where train/val/unknown_val will be written.",
    )
    fathomnet_parser.add_argument(
        "--limit-per-concept",
        type=int,
        default=100,
        help="Maximum number of crops to write per queried concept.",
    )
    fathomnet_parser.add_argument(
        "--page-size",
        type=int,
        default=100,
        help="FathomNet query page size.",
    )
    fathomnet_parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Deterministic validation split fraction for known classes.",
    )
    fathomnet_parser.add_argument(
        "--padding-fraction",
        type=float,
        default=0.1,
        help="Context padding around each bounding box before cropping.",
    )
    fathomnet_parser.add_argument(
        "--no-square-crops",
        action="store_true",
        help="Do not force crops to square boxes.",
    )
    fathomnet_parser.add_argument(
        "--min-box-size",
        type=int,
        default=32,
        help="Skip boxes whose width or height is below this pixel size.",
    )
    fathomnet_parser.add_argument(
        "--dataset-role",
        choices=["known", "unknown"],
        default="known",
        help="Write labeled train/val crops or flat unknown_val crops.",
    )
    fathomnet_parser.add_argument(
        "--include-descendants",
        action="store_true",
        help="Expand each queried concept through a FathomNet taxa provider and ingest descendant labels.",
    )
    fathomnet_parser.add_argument(
        "--taxa-provider",
        default=None,
        help="FathomNet taxa provider name required when --include-descendants is used.",
    )
    fathomnet_parser.add_argument(
        "--review-states",
        default="VERIFIED",
        help="Comma-separated box review states to keep, for example VERIFIED or VERIFIED,UNVERIFIED.",
    )
    fathomnet_parser.add_argument(
        "--min-depth",
        type=float,
        default=None,
        help="Optional minimum depth filter in meters.",
    )
    fathomnet_parser.add_argument(
        "--max-depth",
        type=float,
        default=None,
        help="Optional maximum depth filter in meters.",
    )
    fathomnet_parser.add_argument(
        "--imaging-types",
        default=None,
        help="Optional comma-separated imaging types to keep.",
    )
    fathomnet_parser.add_argument(
        "--cache-dir",
        default=".cache/ingest/images/fathomnet",
        help="Cache directory for downloaded source images.",
    )
    return parser



def main() -> None:
    args = build_parser().parse_args()
    if args.source == "fathomnet":
        concepts = load_concepts(args)
        if not concepts:
            raise SystemExit("At least one concept is required")
        config = FathomNetIngestConfig(
            concepts=concepts,
            output_dir=Path(args.output_dir),
            limit_per_concept=args.limit_per_concept,
            page_size=args.page_size,
            val_fraction=args.val_fraction,
            padding_fraction=args.padding_fraction,
            square_crops=not args.no_square_crops,
            min_box_size=args.min_box_size,
            dataset_role=args.dataset_role,
            cache_dir=Path(args.cache_dir),
            include_descendants=args.include_descendants,
            taxa_provider=args.taxa_provider,
            review_states=parse_csv(args.review_states),
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            imaging_types=parse_csv(args.imaging_types) if args.imaging_types else [],
        )
        result = ingest_fathomnet(config)
        print(f"Output dir: {result['output_dir']}")
        print(f"Manifest: {result['manifest']}")
        print(f"Total crops written: {result['total_crops_written']}")
        for concept_summary in result["concept_summaries"]:
            matched = ", ".join(concept_summary["matched_concepts"])
            print(
                f"  {concept_summary['query_concept']}: "
                f"{concept_summary['crops_written']} crops "
                f"from {concept_summary['images_considered']} images "
                f"matched={matched}"
            )


if __name__ == "__main__":
    main()
