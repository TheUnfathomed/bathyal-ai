"""Grid sweep over LoRA fine-tuning hyperparameters."""

from __future__ import annotations

import argparse
import itertools
import json
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

from bathyal_ai.finetune import FinetuneConfig, run_finetune


def ParseFloatList(raw: str) -> list[float]:
    return [float(v.strip()) for v in raw.split(",") if v.strip()]


def ParseIntList(raw: str) -> list[int]:
    return [int(v.strip()) for v in raw.split(",") if v.strip()]


@dataclass(slots=True)
class SweepConfig:
    train_dir: Path
    val_dir: Path
    output_root: Path = Path("artifacts/sweep")
    unknown_dir: Path | None = None
    bioclip_model: str = "hf-hub:imageomics/bioclip-2"
    ranks: list[int] = field(default_factory=lambda: [8, 16, 32])
    lrs: list[float] = field(default_factory=lambda: [5e-5, 1e-4, 2e-4])
    target_sets: list[list[str]] = field(default_factory=lambda: [["q", "k", "v", "o"]])
    dropouts: list[float] = field(default_factory=lambda: [0.0, 0.05, 0.1])
    epochs: int = 20
    batch_size: int = 32
    min_examples_per_label: int = 2
    device: str = "auto"
    seed: int = 13


def RunSweep(sweep: SweepConfig) -> list[dict]:
    Combinations = list(itertools.product(sweep.ranks, sweep.lrs, sweep.target_sets, sweep.dropouts))
    print(f"Sweep: {len(Combinations)} configurations\n")

    Results: list[dict] = []

    for i, (rank, lr, targets, dropout) in enumerate(Combinations):
        TargetLabel = "+".join(targets)
        RunName = f"r{rank}_lr{lr:.0e}_t-{TargetLabel}_d{dropout}"
        OutputDir = sweep.output_root / RunName
        print(f"[{i + 1}/{len(Combinations)}] {RunName}")

        Config = FinetuneConfig(
            train_dir=sweep.train_dir,
            val_dir=sweep.val_dir,
            output_dir=OutputDir,
            unknown_dir=sweep.unknown_dir,
            bioclip_model=sweep.bioclip_model,
            lora_rank=rank,
            lora_alpha=float(rank * 2),
            lora_targets=targets,
            lora_dropout=dropout,
            lora_lr=lr,
            epochs=sweep.epochs,
            batch_size=sweep.batch_size,
            min_examples_per_label=sweep.min_examples_per_label,
            device=sweep.device,
            seed=sweep.seed,
        )

        StartTime = time.time()
        try:
            Report = run_finetune(Config)
            Elapsed = time.time() - StartTime
            Summary = Report["training_summary"]
            Results.append({
                "run_name": RunName,
                "rank": rank,
                "lr": lr,
                "targets": targets,
                "dropout": dropout,
                "val_accuracy": Summary["head_training"]["best_val_accuracy"],
                "val_loss": Summary["head_training"]["best_val_loss"],
                "final_val_top1": Summary["final_val_top1"],
                "epochs_trained": Summary["head_training"]["epochs_trained"],
                "lora_param_count": Summary["lora_param_count"],
                "elapsed_seconds": round(Elapsed, 1),
                "status": "ok",
            })
        except Exception as e:
            Elapsed = time.time() - StartTime
            Results.append({
                "run_name": RunName,
                "rank": rank,
                "lr": lr,
                "targets": targets,
                "dropout": dropout,
                "status": "error",
                "error": str(e),
                "elapsed_seconds": round(Elapsed, 1),
            })
            traceback.print_exc()

    return Results


def PrintResultsTable(results: list[dict]) -> None:
    OkResults = [r for r in results if r["status"] == "ok"]
    if not OkResults:
        print("No successful runs.")
        return

    OkResults.sort(key=lambda r: r["final_val_top1"], reverse=True)
    Header = f"{'Run':<45s}  {'Rank':>4}  {'LR':>8}  {'Drop':>5}  {'ValAcc':>7}  {'Top1':>6}  {'Epochs':>6}  {'Time':>8}"
    print(f"\n{'=' * len(Header)}")
    print(Header)
    print(f"{'-' * len(Header)}")
    for r in OkResults:
        print(
            f"{r['run_name']:<45s}  {r['rank']:>4}  {r['lr']:>8.1e}  {r['dropout']:>5.2f}  "
            f"{r['val_accuracy']:>7.4f}  {r['final_val_top1']:>6.4f}  "
            f"{r['epochs_trained']:>6}  {r['elapsed_seconds']:>7.1f}s"
        )
    print(f"{'=' * len(Header)}")

    Failed = [r for r in results if r["status"] == "error"]
    if Failed:
        print(f"\n{len(Failed)} run(s) failed:")
        for r in Failed:
            print(f"  {r['run_name']}: {r['error']}")


def BuildParser() -> argparse.ArgumentParser:
    Parser = argparse.ArgumentParser(description="Grid sweep over LoRA hyperparameters.")
    Parser.add_argument("--train-dir", type=Path, required=True)
    Parser.add_argument("--val-dir", type=Path, required=True)
    Parser.add_argument("--unknown-dir", type=Path, default=None)
    Parser.add_argument("--output-root", type=Path, default=Path("artifacts/sweep"))
    Parser.add_argument("--bioclip-model", default="hf-hub:imageomics/bioclip-2")
    Parser.add_argument("--ranks", type=str, default="8,16,32")
    Parser.add_argument("--lrs", type=str, default="5e-5,1e-4,2e-4")
    Parser.add_argument("--dropouts", type=str, default="0.0,0.05,0.1")
    Parser.add_argument("--include-mlp", action="store_true", help="Include attention+MLP target set in sweep.")
    Parser.add_argument("--epochs", type=int, default=20)
    Parser.add_argument("--batch-size", type=int, default=32)
    Parser.add_argument("--min-examples-per-label", type=int, default=2)
    Parser.add_argument("--device", default="auto")
    Parser.add_argument("--seed", type=int, default=13)
    return Parser


def main() -> None:
    Args = BuildParser().parse_args()

    TargetSets: list[list[str]] = [["q", "k", "v", "o"]]
    if Args.include_mlp:
        TargetSets.append(["q", "k", "v", "o", "mlp_fc", "mlp_proj"])

    Sweep = SweepConfig(
        train_dir=Args.train_dir,
        val_dir=Args.val_dir,
        output_root=Args.output_root,
        unknown_dir=Args.unknown_dir,
        bioclip_model=Args.bioclip_model,
        ranks=ParseIntList(Args.ranks),
        lrs=ParseFloatList(Args.lrs),
        target_sets=TargetSets,
        dropouts=ParseFloatList(Args.dropouts),
        epochs=Args.epochs,
        batch_size=Args.batch_size,
        min_examples_per_label=Args.min_examples_per_label,
        device=Args.device,
        seed=Args.seed,
    )

    Results = RunSweep(Sweep)
    PrintResultsTable(Results)

    Sweep.output_root.mkdir(parents=True, exist_ok=True)
    ResultsPath = Sweep.output_root / "sweep_results.json"
    ResultsPath.write_text(json.dumps(Results, indent=2, default=str), encoding="utf-8")
    print(f"\nResults saved to {ResultsPath}")


if __name__ == "__main__":
    main()
