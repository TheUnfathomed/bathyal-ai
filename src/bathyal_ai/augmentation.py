"""Underwater image augmentation pipeline for LoRA fine-tuning."""

from __future__ import annotations

import torchvision.transforms as T


OPENCLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENCLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def extract_normalize_params(preprocess) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if hasattr(preprocess, "transforms"):
        for transform in preprocess.transforms:
            if isinstance(transform, T.Normalize):
                return tuple(float(v) for v in transform.mean), tuple(float(v) for v in transform.std)
    return OPENCLIP_MEAN, OPENCLIP_STD


def build_train_transform(
    image_size: int = 224,
    mean: tuple[float, ...] = OPENCLIP_MEAN,
    std: tuple[float, ...] = OPENCLIP_STD,
) -> T.Compose:
    return T.Compose([
        T.RandomResizedCrop(image_size, scale=(0.7, 1.0), ratio=(0.9, 1.1), interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.4, hue=0.08),
        T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 1.5))], p=0.3),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])


def build_val_transform(
    image_size: int = 224,
    mean: tuple[float, ...] = OPENCLIP_MEAN,
    std: tuple[float, ...] = OPENCLIP_STD,
) -> T.Compose:
    return T.Compose([
        T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])
