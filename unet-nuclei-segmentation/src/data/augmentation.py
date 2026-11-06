"""Transformasi Albumentations: resize, pad, augmentasi, normalisasi."""

from __future__ import annotations

import cv2
from albumentations import (
    Affine,
    Compose,
    GaussianBlur,
    GaussNoise,
    HorizontalFlip,
    LongestMaxSize,
    Normalize,
    PadIfNeeded,
    RandomBrightnessContrast,
    RandomRotate90,
    VerticalFlip,
)
from albumentations.pytorch import ToTensorV2


def build_train_transforms(
    image_size: int,
    mean: tuple[float, float, float] = (0.5, 0.5, 0.5),
    std: tuple[float, float, float] = (0.5, 0.5, 0.5),
) -> Compose:
    """Transformasi training: jaga aspek rasio, pad ke persegi, augmentasi."""
    return Compose(
        [
            LongestMaxSize(
                max_size=image_size,
                interpolation=cv2.INTER_AREA,
                mask_interpolation=cv2.INTER_NEAREST,
            ),
            PadIfNeeded(
                min_height=image_size,
                min_width=image_size,
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
                fill_mask=0,
            ),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            RandomRotate90(p=0.5),
            Affine(
                scale=(0.9, 1.1),
                translate_percent=(-0.05, 0.05),
                rotate=(-15, 15),
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
                fill_mask=0,
                p=0.5,
            ),
            RandomBrightnessContrast(
                brightness_limit=0.15,
                contrast_limit=0.15,
                p=0.3,
            ),
            GaussNoise(p=0.2),
            GaussianBlur(blur_limit=(3, 5), p=0.1),
            Normalize(mean=mean, std=std, max_pixel_value=255.0),
            ToTensorV2(),
        ]
    )


def build_eval_transforms(
    image_size: int,
    mean: tuple[float, float, float] = (0.5, 0.5, 0.5),
    std: tuple[float, float, float] = (0.5, 0.5, 0.5),
) -> Compose:
    """Transformasi validasi/test tanpa augmentasi acak."""
    return Compose(
        [
            LongestMaxSize(
                max_size=image_size,
                interpolation=cv2.INTER_AREA,
                mask_interpolation=cv2.INTER_NEAREST,
            ),
            PadIfNeeded(
                min_height=image_size,
                min_width=image_size,
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
                fill_mask=0,
            ),
            Normalize(mean=mean, std=std, max_pixel_value=255.0),
            ToTensorV2(),
        ]
    )
