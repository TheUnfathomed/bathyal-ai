"""Pre-resolve FathomNet taxa descendants in parallel, then run ingest with expanded species list."""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from fathomnet.api import taxa


def ResolveTaxon(concept: str, provider: str) -> tuple[str, list[str]]:
    try:
        descendants = taxa.find_taxa(provider, concept)
        species = [
            entry.name for entry in descendants
            if entry.name and entry.rank and entry.rank.lower() == "species"
        ]
        if not species:
            species = [entry.name for entry in descendants if entry.name and " " in entry.name]
        if not species and " " in concept:
            species = [concept]
        return concept, species
    except Exception as e:
        if " " in concept:
            return concept, [concept]
        return concept, []


def main() -> None:
    import argparse

    Parser = argparse.ArgumentParser(description="Resolve FathomNet taxa descendants in parallel.")
    Parser.add_argument("--concepts-file", type=Path, required=True)
    Parser.add_argument("--output-file", type=Path, required=True)
    Parser.add_argument("--taxa-provider", type=str, default="fathomnet")
    Parser.add_argument("--workers", type=int, default=16)
    Args = Parser.parse_args()

    Concepts = [line.strip() for line in Args.concepts_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"Resolving descendants for {len(Concepts)} concepts with {Args.workers} workers...")

    AllSpecies: set[str] = set()
    Resolved = 0

    with ThreadPoolExecutor(max_workers=Args.workers) as pool:
        Futures = {pool.submit(ResolveTaxon, c, Args.taxa_provider): c for c in Concepts}
        for future in as_completed(Futures):
            concept, species = future.result()
            AllSpecies.update(species)
            Resolved += 1
            if Resolved % 50 == 0:
                print(f"  Resolved {Resolved}/{len(Concepts)} concepts, {len(AllSpecies)} unique species so far")

    print(f"\nResolved {len(Concepts)} concepts -> {len(AllSpecies)} unique species")
    Args.output_file.parent.mkdir(parents=True, exist_ok=True)
    Args.output_file.write_text("\n".join(sorted(AllSpecies)) + "\n", encoding="utf-8")
    print(f"Wrote {Args.output_file}")


if __name__ == "__main__":
    main()
