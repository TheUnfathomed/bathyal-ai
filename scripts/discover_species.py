"""Query FathomNet for the most well-represented biological taxa and write a concepts file."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from fathomnet.api import boundingboxes

NON_BIOLOGICAL_PATTERNS = [
    r"\bequipment\b", r"\bROV\b", r"\blaser\b", r"\bcable\b", r"\bpipe\b",
    r"\banchor\b", r"\binstrument\b", r"\bframe\b", r"\bweight\b", r"\btool\b",
    r"\bdebris\b", r"\btrash\b", r"\blitter\b", r"\bplastic\b", r"\bbag\b",
    r"\bbottle\b", r"\bcan\b", r"\brope\b", r"\bnet\b", r"\bline\b",
    r"\bsubstrate\b", r"\bsediment\b", r"\bsand\b", r"\bboulder\b",
    r"\bcobble\b", r"\bmud\b", r"\brock\b", r"\bgravel\b",
    r"\bunknown\b", r"\bunidentified\b", r"\bnone\b", r"\bother\b",
    r"\bobject\b", r"\bvehicle\b", r"\bship\b", r"\bboat\b",
    r"\bplatform\b", r"\bmooring\b", r"\bsampler\b", r"\bcamera\b",
    r"\blight\b", r"\btube\b", r"\bwire\b", r"\bchain\b",
    r"\bbubble\b", r"\bparticle\b", r"\bmarine snow\b",
]
NON_BIOLOGICAL_RE = re.compile("|".join(NON_BIOLOGICAL_PATTERNS), re.IGNORECASE)


GENERIC_TAXA = {
    "marine organism", "bony fish", "sea fan", "brittle star", "feather star",
    "sea star", "sea cucumber", "sea urchin", "sea pen", "sea whip",
    "soft coral", "hard coral", "stony coral", "black coral",
}


def IsLikelyBiological(concept: str) -> bool:
    if not concept or not concept.strip():
        return False
    if NON_BIOLOGICAL_RE.search(concept):
        return False
    if concept.lower() in GENERIC_TAXA:
        return False
    if not concept[0].isupper():
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Discover well-represented FathomNet taxa.")
    parser.add_argument("--min-count", type=int, default=100, help="Minimum bounding box count per concept.")
    parser.add_argument("--top-n", type=int, default=30, help="Number of top concepts to output.")
    parser.add_argument("--output", type=str, default="scripts/concepts.txt", help="Output concepts file path.")
    args = parser.parse_args()

    print("Querying FathomNet for bounding box counts per concept...")
    counts = boundingboxes.count_total_by_concept()
    print(f"  Total concepts returned: {len(counts)}")

    biological = [
        entry for entry in counts
        if entry.concept and entry.count and entry.count >= args.min_count and IsLikelyBiological(entry.concept)
    ]
    biological.sort(key=lambda e: e.count, reverse=True)
    selected = biological[: args.top_n]

    print(f"\nTop {len(selected)} biological taxa (min {args.min_count} annotations):\n")
    for i, entry in enumerate(selected, 1):
        print(f"  {i:3d}. {entry.concept:<40s}  ({entry.count:,} boxes)")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(entry.concept for entry in selected) + "\n",
        encoding="utf-8",
    )
    print(f"\nWrote {len(selected)} concepts to {output_path}")


if __name__ == "__main__":
    main()
