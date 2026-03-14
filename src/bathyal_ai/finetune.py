"""LoRA fine-tuning pipeline for BioCLIP 2 species classifier."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .augmentation import build_train_transform, build_val_transform, extract_normalize_params
from .classifier import LinearClassifierHead, SpeciesClassifierBundle, ThresholdConfig
from .data import IndexedDataset, LabeledExample, ensure_labels_subset, filter_dataset_by_min_examples, index_labeled_dataset, list_images
from .embeddings import BioClipEmbedder
from .lora import LoraConfig, apply_lora_to_vision_encoder, load_lora_state_dict, lora_state_dict, merge_lora_weights
from .training import build_centroids, compute_class_weights, encode_labels, fit_temperature, select_thresholds, set_seed


@dataclass(slots=True)
class FinetuneConfig:
    train_dir: Path
    val_dir: Path
    output_dir: Path = Path("artifacts/species_classifier_lora")
    unknown_dir: Path | None = None
    bioclip_model: str = "hf-hub:imageomics/bioclip-2"
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_target_blocks: list[int] | None = None
    lora_targets: list[str] = field(default_factory=lambda: ["q", "k", "v", "o"])
    lora_lr: float = 1e-4
    head_lr: float = 5e-3
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05
    epochs: int = 20
    batch_size: int = 32
    warmup_epochs: int = 2
    min_examples_per_label: int = 2
    target_precision: float = 0.98
    early_stopping_patience: int = 5
    mixed_precision: bool = True
    num_workers: int = 4
    seed: int = 13
    device: str = "auto"
    top_candidate_count: int = 5
    probability_thresholds: list[float] = field(
        default_factory=lambda: [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    )
    margin_thresholds: list[float] = field(default_factory=lambda: [0.0, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.25])
    centroid_thresholds: list[float] = field(default_factory=lambda: [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    resume_from: Path | None = None
    resume_checkpoint: Path | None = None
    checkpoint_every: int | None = None


@dataclass(slots=True)
class PriorRun:
    labels: list[str]
    lora_config: LoraConfig
    lora_state: dict[str, torch.Tensor]
    head_state: dict[str, torch.Tensor]
    centroids: np.ndarray
    bioclip_model: str
    feature_dim: int


def load_prior_run(resume_dir: Path) -> PriorRun:
    required_files = ["metadata.json", "lora_weights.pt", "head.pt", "centroids.npy"]
    missing = [f for f in required_files if not (resume_dir / f).exists()]
    if missing:
        raise FileNotFoundError(f"Prior run at {resume_dir} is missing: {', '.join(missing)}")

    metadata = json.loads((resume_dir / "metadata.json").read_text(encoding="utf-8"))
    lora_config = LoraConfig.from_dict(metadata.get("lora_config", {}))
    lora_state = torch.load(resume_dir / "lora_weights.pt", map_location="cpu", weights_only=True)
    head_data = torch.load(resume_dir / "head.pt", map_location="cpu", weights_only=True)
    head_state = head_data["state_dict"]
    centroids = np.load(resume_dir / "centroids.npy")

    return PriorRun(
        labels=metadata["labels"],
        lora_config=lora_config,
        lora_state=lora_state,
        head_state=head_state,
        centroids=centroids,
        bioclip_model=metadata["bioclip_model"],
        feature_dim=metadata["feature_dim"],
    )


def expand_head_for_new_labels(
    prior_head_state: dict[str, torch.Tensor],
    prior_labels: list[str],
    merged_labels: list[str],
    feature_dim: int,
) -> dict[str, torch.Tensor]:
    new_head = LinearClassifierHead(feature_dim=feature_dim, class_count=len(merged_labels))
    new_state = new_head.state_dict()

    prior_index = {label: i for i, label in enumerate(prior_labels)}
    w_old = prior_head_state["linear.weight"]
    b_old = prior_head_state["linear.bias"]

    for j, label in enumerate(merged_labels):
        if label in prior_index:
            i = prior_index[label]
            new_state["linear.weight"][j] = w_old[i]
            new_state["linear.bias"][j] = b_old[i]

    return new_state


def build_centroids_with_prior(
    train_embeddings: np.ndarray,
    train_targets: np.ndarray,
    class_count: int,
    prior_centroids: np.ndarray,
    prior_labels: list[str],
    merged_labels: list[str],
) -> np.ndarray:
    prior_index = {label: i for i, label in enumerate(prior_labels)}
    centroids: list[np.ndarray] = []
    for class_idx in range(class_count):
        class_embeddings = train_embeddings[train_targets == class_idx]
        if len(class_embeddings) > 0:
            centroid = class_embeddings.mean(axis=0)
            norm = max(float(np.linalg.norm(centroid)), 1e-12)
            centroids.append((centroid / norm).astype(np.float32))
        else:
            label = merged_labels[class_idx]
            if label not in prior_index:
                raise ValueError(f"Class '{label}' has no training embeddings and is not in the prior run")
            old_idx = prior_index[label]
            centroids.append(prior_centroids[old_idx])
    return np.vstack(centroids)


def save_checkpoint(
    path: Path,
    epoch: int,
    model: nn.Module,
    head: LinearClassifierHead,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    best_val_accuracy: float,
    best_val_loss: float,
    best_lora_state: dict[str, torch.Tensor],
    best_head_state: dict[str, torch.Tensor],
    epochs_without_improvement: int,
    labels: list[str],
    bioclip_model: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "best_val_accuracy": best_val_accuracy,
        "best_val_loss": best_val_loss,
        "best_lora_state": best_lora_state,
        "best_head_state": best_head_state,
        "current_lora_state": lora_state_dict(model),
        "current_head_state": {k: v.clone() for k, v in head.state_dict().items()},
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "epochs_without_improvement": epochs_without_improvement,
        "labels": labels,
        "bioclip_model": bioclip_model,
    }, path)


def load_checkpoint(path: Path) -> dict[str, object]:
    return torch.load(path, map_location="cpu", weights_only=True)


class SpeciesImageDataset(Dataset):
    def __init__(self, dataset: IndexedDataset, label_to_index: dict[str, int] | None, transform) -> None:
        self.examples = dataset.examples
        self.label_to_index = label_to_index
        self.transform = transform

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int] | torch.Tensor:
        example = self.examples[index]
        with Image.open(example.path) as img:
            image = img.convert("RGB")
        tensor = self.transform(image)
        if self.label_to_index is not None:
            return tensor, self.label_to_index[example.label]
        return tensor


def _embed_dataset_through_model(
    model: nn.Module,
    dataset: IndexedDataset,
    transform,
    device: str,
    batch_size: int = 64,
    num_workers: int = 2,
) -> np.ndarray:
    loader = DataLoader(
        SpeciesImageDataset(dataset, None, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device != "cpu",
    )
    all_embeddings: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for batch_tensors in loader:
            batch_tensors = batch_tensors.to(device)
            embeddings = model.encode_image(batch_tensors)
            embeddings = F.normalize(embeddings, dim=-1)
            all_embeddings.append(embeddings.cpu().numpy().astype(np.float32))
    return np.concatenate(all_embeddings, axis=0)


def run_finetune(config: FinetuneConfig) -> dict[str, object]:
    set_seed(config.seed)
    device = BioClipEmbedder._resolve_device(config.device)

    raw_train_dataset = index_labeled_dataset(config.train_dir)
    train_dataset = filter_dataset_by_min_examples(raw_train_dataset, config.min_examples_per_label)
    if train_dataset.example_count == 0:
        raise ValueError(f"No training labels remain after min_examples_per_label={config.min_examples_per_label}")
    dropped_train_labels = sorted(set(raw_train_dataset.labels) - set(train_dataset.labels))

    val_dataset = index_labeled_dataset(config.val_dir)
    unknown_paths = list_images(config.unknown_dir) if config.unknown_dir is not None else []

    prior_run: PriorRun | None = None
    retained_only_labels: list[str] = []
    new_labels: list[str] = []

    if config.resume_from is not None:
        prior_run = load_prior_run(config.resume_from)
        if config.bioclip_model != prior_run.bioclip_model:
            raise ValueError(
                f"BioCLIP model mismatch: current run uses '{config.bioclip_model}' "
                f"but prior run used '{prior_run.bioclip_model}'"
            )
        labels = sorted(set(prior_run.labels) | set(train_dataset.labels))
        retained_only_labels = sorted(set(prior_run.labels) - set(train_dataset.labels))
        new_labels = sorted(set(train_dataset.labels) - set(prior_run.labels))
        if retained_only_labels:
            print(f"Retaining {len(retained_only_labels)} prior classes not in new training data")
        if new_labels:
            print(f"Adding {len(new_labels)} new classes")
    else:
        labels = train_dataset.labels

    label_to_index = {label: index for index, label in enumerate(labels)}
    class_count = len(labels)

    ensure_labels_subset(val_dataset, labels, "Validation dataset")

    print(f"Training: {train_dataset.example_count} examples, {class_count} classes")
    print(f"Validation: {val_dataset.example_count} examples")
    if unknown_paths:
        print(f"Unknown calibration: {len(unknown_paths)} images")

    model, _, preprocess = open_clip.create_model_and_transforms(config.bioclip_model)
    model = model.to(device)

    mean, std = extract_normalize_params(preprocess)
    image_size = 224
    train_transform = build_train_transform(image_size, mean, std)
    val_transform = build_val_transform(image_size, mean, std)

    lora_config = LoraConfig(
        rank=config.lora_rank,
        alpha=config.lora_alpha,
        target_blocks=config.lora_target_blocks,
        targets=config.lora_targets,
    )
    lora_params = apply_lora_to_vision_encoder(model, lora_config)
    lora_param_count = sum(p.numel() for p in lora_params)
    total_param_count = sum(p.numel() for p in model.parameters())
    print(f"LoRA params: {lora_param_count:,} / {total_param_count:,} total ({100 * lora_param_count / total_param_count:.2f}%)")

    if prior_run is not None:
        print("Loading prior LoRA weights...")
        load_lora_state_dict(model, prior_run.lora_state)

    feature_dim = model.visual.proj.shape[1] if hasattr(model.visual, "proj") and model.visual.proj is not None else model.visual.transformer.width
    head = LinearClassifierHead(feature_dim=feature_dim, class_count=class_count).to(device)

    if prior_run is not None:
        expanded_state = expand_head_for_new_labels(
            prior_run.head_state, prior_run.labels, labels, feature_dim,
        )
        head.load_state_dict(expanded_state)
        head = head.to(device)

    train_ds = SpeciesImageDataset(train_dataset, label_to_index, train_transform)
    val_ds = SpeciesImageDataset(val_dataset, label_to_index, val_transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=device != "cpu",
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=device != "cpu",
    )

    class_weights = compute_class_weights(
        encode_labels([e.label for e in train_dataset.examples], label_to_index), class_count
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=config.label_smoothing)

    optimizer = torch.optim.AdamW([
        {"params": lora_params, "lr": config.lora_lr, "weight_decay": config.weight_decay},
        {"params": head.parameters(), "lr": config.head_lr, "weight_decay": config.weight_decay},
    ])

    total_steps = config.epochs * len(train_loader)
    warmup_steps = config.warmup_epochs * len(train_loader)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    use_amp = config.mixed_precision and device != "cpu"
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)

    best_val_accuracy = 0.0
    best_val_loss = float("inf")
    best_lora_state = lora_state_dict(model)
    best_head_state = {k: v.clone() for k, v in head.state_dict().items()}
    epochs_without_improvement = 0
    epochs_completed = 0
    start_epoch = 0

    if config.resume_checkpoint is not None:
        print(f"Resuming from checkpoint {config.resume_checkpoint}...")
        ckpt = load_checkpoint(config.resume_checkpoint)
        if ckpt["labels"] != labels:
            raise ValueError("Checkpoint label set does not match current training labels")
        if ckpt["bioclip_model"] != config.bioclip_model:
            raise ValueError("Checkpoint BioCLIP model does not match current config")
        load_lora_state_dict(model, ckpt["current_lora_state"])
        head.load_state_dict(ckpt["current_head_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        scaler.load_state_dict(ckpt["scaler_state"])
        start_epoch = ckpt["epoch"]
        best_val_accuracy = ckpt["best_val_accuracy"]
        best_val_loss = ckpt["best_val_loss"]
        best_lora_state = ckpt["best_lora_state"]
        best_head_state = ckpt["best_head_state"]
        epochs_without_improvement = ckpt["epochs_without_improvement"]
        print(f"  Resumed at epoch {start_epoch}, best_val_acc={best_val_accuracy:.3f}")

    for epoch in range(start_epoch, config.epochs):
        model.train()
        head.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        for batch_images, batch_targets in train_loader:
            batch_images = batch_images.to(device)
            batch_targets = batch_targets.to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                embeddings = model.encode_image(batch_images)
                embeddings = F.normalize(embeddings, dim=-1)
                logits = head(embeddings.float())
                loss = criterion(logits, batch_targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            scheduler.step()

            epoch_loss += loss.item() * batch_targets.size(0)
            epoch_correct += (logits.argmax(dim=1) == batch_targets).sum().item()
            epoch_total += batch_targets.size(0)

        train_loss = epoch_loss / max(epoch_total, 1)
        train_accuracy = epoch_correct / max(epoch_total, 1)

        model.eval()
        head.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.inference_mode():
            for batch_images, batch_targets in val_loader:
                batch_images = batch_images.to(device)
                batch_targets = batch_targets.to(device)

                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                    embeddings = model.encode_image(batch_images)
                    embeddings = F.normalize(embeddings, dim=-1)
                    logits = head(embeddings.float())
                    loss = criterion(logits, batch_targets)

                val_loss += loss.item() * batch_targets.size(0)
                val_correct += (logits.argmax(dim=1) == batch_targets).sum().item()
                val_total += batch_targets.size(0)

        val_loss_avg = val_loss / max(val_total, 1)
        val_accuracy = val_correct / max(val_total, 1)

        lr_current = optimizer.param_groups[0]["lr"]
        print(
            f"  Epoch {epoch + 1}/{config.epochs}: "
            f"train_loss={train_loss:.4f} train_acc={train_accuracy:.3f} "
            f"val_loss={val_loss_avg:.4f} val_acc={val_accuracy:.3f} "
            f"lr={lr_current:.2e}"
        )

        epochs_completed = epoch + 1
        is_best = val_accuracy > best_val_accuracy or (val_accuracy == best_val_accuracy and val_loss_avg < best_val_loss)
        if is_best:
            best_val_accuracy = val_accuracy
            best_val_loss = val_loss_avg
            best_lora_state = lora_state_dict(model)
            best_head_state = {k: v.clone() for k, v in head.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.early_stopping_patience:
                print(f"  Early stopping at epoch {epoch + 1}")
                break

        if config.checkpoint_every is not None:
            periodic = (epoch + 1) % config.checkpoint_every == 0
            if periodic or is_best:
                checkpoint_dir = config.output_dir / "checkpoints"
                ckpt_args = dict(
                    model=model, head=head, optimizer=optimizer, scheduler=scheduler,
                    scaler=scaler, best_val_accuracy=best_val_accuracy,
                    best_val_loss=best_val_loss, best_lora_state=best_lora_state,
                    best_head_state=best_head_state,
                    epochs_without_improvement=epochs_without_improvement,
                    labels=labels, bioclip_model=config.bioclip_model,
                )
                primary_path = checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pt" if periodic else checkpoint_dir / "checkpoint_best.pt"
                save_checkpoint(path=primary_path, epoch=epoch + 1, **ckpt_args)
                if periodic and is_best:
                    import shutil
                    shutil.copy2(primary_path, checkpoint_dir / "checkpoint_best.pt")

    load_lora_state_dict(model, best_lora_state)
    head.load_state_dict(best_head_state)
    model.eval()
    head.eval()

    print("Re-embedding datasets through fine-tuned encoder...")
    train_embeddings = _embed_dataset_through_model(model, train_dataset, val_transform, device, batch_size=config.batch_size)
    val_embeddings = _embed_dataset_through_model(model, val_dataset, val_transform, device, batch_size=config.batch_size)

    unknown_embeddings = None
    if unknown_paths:
        unknown_dataset = IndexedDataset(
            root=config.unknown_dir,
            examples=[LabeledExample(path=p, label="__unlabeled__") for p in unknown_paths],
        )
        unknown_embeddings = _embed_dataset_through_model(model, unknown_dataset, val_transform, device, batch_size=config.batch_size)

    train_targets = encode_labels([e.label for e in train_dataset.examples], label_to_index)
    val_targets = encode_labels([e.label for e in val_dataset.examples], label_to_index)

    print("Calibrating temperature...")
    temperature = fit_temperature(head, val_embeddings, val_targets, device)

    print("Building centroids...")
    if prior_run is not None and retained_only_labels:
        centroids = build_centroids_with_prior(
            train_embeddings, train_targets, class_count,
            prior_run.centroids, prior_run.labels, labels,
        )
    else:
        centroids = build_centroids(train_embeddings, train_targets, class_count=class_count)

    saved_lora_state = lora_state_dict(model)
    saved_lora_config = {
        "rank": config.lora_rank,
        "alpha": config.lora_alpha,
        "target_blocks": config.lora_target_blocks,
        "targets": config.lora_targets,
    }

    merge_lora_weights(model)
    model.eval()

    embedder = BioClipEmbedder.from_model(
        model_name=config.bioclip_model,
        model=model,
        preprocess=preprocess,
        device=device,
        batch_size=config.batch_size,
    )

    bundle = SpeciesClassifierBundle(
        bioclip_model=config.bioclip_model,
        labels=labels,
        head=head,
        centroids=centroids,
        temperature=temperature,
        thresholds=ThresholdConfig(probability=0.0, margin=0.0, centroid_similarity=0.0),
        device=device,
        embed_batch_size=config.batch_size,
        top_candidate_count=config.top_candidate_count,
        lora_config=saved_lora_config,
    )
    bundle.embedder = embedder

    from .training import TrainingConfig
    threshold_config = TrainingConfig(
        train_dir=config.train_dir,
        val_dir=config.val_dir,
        probability_thresholds=config.probability_thresholds,
        margin_thresholds=config.margin_thresholds,
        centroid_thresholds=config.centroid_thresholds,
        target_precision=config.target_precision,
    )

    print("Selecting thresholds...")
    thresholds, best_threshold_summary, threshold_leaderboard = select_thresholds(
        bundle=bundle,
        val_embeddings=val_embeddings,
        val_targets=val_targets,
        unknown_embeddings=unknown_embeddings,
        config=threshold_config,
    )
    bundle.thresholds = thresholds

    final_val_top1 = float(np.mean(bundle.predict_arrays(val_embeddings)["top_indices"] == val_targets)) if len(val_targets) else 0.0

    training_summary = {
        "train_examples": train_dataset.example_count,
        "val_examples": val_dataset.example_count,
        "unknown_examples": len(unknown_paths),
        "label_count": class_count,
        "labels": labels,
        "dropped_train_labels": dropped_train_labels,
        "lora_config": saved_lora_config,
        "lora_param_count": lora_param_count,
        "head_training": {
            "best_val_loss": round(best_val_loss, 6),
            "best_val_accuracy": round(best_val_accuracy, 6),
            "epochs_trained": epochs_completed,
        },
        "temperature": round(temperature, 6),
        "threshold_selection_target_precision": config.target_precision,
        "selected_threshold_summary": best_threshold_summary,
        "threshold_leaderboard": threshold_leaderboard,
        "final_val_top1": round(final_val_top1, 6),
    }
    if config.resume_from is not None:
        training_summary["resumed_from"] = str(config.resume_from)
        training_summary["retained_only_labels"] = retained_only_labels
        training_summary["new_labels"] = new_labels
    bundle.training_summary = training_summary

    config.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(saved_lora_state, config.output_dir / "lora_weights.pt")
    bundle.save(config.output_dir)

    report = {
        "output_dir": str(config.output_dir),
        "classifier_bundle": str(config.output_dir),
        "bioclip_model": config.bioclip_model,
        "device": device,
        "thresholds": thresholds.to_dict(),
        "training_summary": training_summary,
    }
    (config.output_dir / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Bundle saved to {config.output_dir}")
    print(f"Final val top1: {final_val_top1:.3f}")
    return report
