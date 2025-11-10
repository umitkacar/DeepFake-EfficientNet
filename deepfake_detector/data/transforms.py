"""
Data augmentation and preprocessing transforms.
Uses albumentations for efficient image transformations.
"""

from albumentations import (
    Compose,
    HorizontalFlip,
    RandomResizedCrop,
    Resize,
    Normalize,
    VerticalFlip,
    Rotate,
    ShiftScaleRotate,
    OpticalDistortion,
    GridDistortion,
    ElasticTransform,
    JpegCompression,
    HueSaturationValue,
    RGBShift,
    RandomBrightness,
    RandomContrast,
    Blur,
    MotionBlur,
    MedianBlur,
    GaussNoise,
    CLAHE,
    RandomGamma,
    CoarseDropout,
)
from albumentations.pytorch import ToTensorV2
from typing import Optional


def get_train_transforms(image_size: int = 224, use_heavy_augmentation: bool = False) -> Compose:
    """
    Get training data augmentation pipeline.

    Args:
        image_size: Target image size
        use_heavy_augmentation: Whether to use aggressive augmentation

    Returns:
        Albumentations Compose object with training transforms

    Example:
        >>> transforms = get_train_transforms(224, use_heavy_augmentation=True)
        >>> augmented = transforms(image=image)
    """
    if use_heavy_augmentation:
        return Compose(
            [
                # Geometric transforms
                HorizontalFlip(p=0.5),
                VerticalFlip(p=0.1),
                RandomResizedCrop(
                    image_size, image_size, scale=(0.5, 1.0), ratio=(0.9, 1.1), p=0.5
                ),
                Rotate(limit=15, p=0.3),
                ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.3),
                # Optical distortions
                OpticalDistortion(distort_limit=0.1, p=0.2),
                GridDistortion(p=0.2),
                ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.2),
                # Compression and noise
                JpegCompression(quality_lower=75, quality_upper=100, p=0.3),
                GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                # Color augmentations
                HueSaturationValue(
                    hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.3
                ),
                RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.3),
                RandomBrightness(limit=0.2, p=0.3),
                RandomContrast(limit=0.2, p=0.3),
                RandomGamma(gamma_limit=(80, 120), p=0.3),
                CLAHE(p=0.2),
                # Blur
                Blur(blur_limit=3, p=0.2),
                MotionBlur(blur_limit=3, p=0.2),
                MedianBlur(blur_limit=3, p=0.2),
                # Cutout
                CoarseDropout(max_holes=8, max_height=32, max_width=32, fill_value=0, p=0.3),
                # Final resize and normalization
                Resize(image_size, image_size, always_apply=True),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], always_apply=True),
                ToTensorV2(),
            ]
        )
    else:
        return Compose(
            [
                HorizontalFlip(p=0.5),
                RandomResizedCrop(image_size, image_size, scale=(0.5, 1.0), p=0.5),
                Resize(image_size, image_size, always_apply=True),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], always_apply=True),
                ToTensorV2(),
            ]
        )


def get_val_transforms(image_size: int = 224) -> Compose:
    """
    Get validation/test data preprocessing pipeline.

    Args:
        image_size: Target image size

    Returns:
        Albumentations Compose object with validation transforms

    Example:
        >>> transforms = get_val_transforms(224)
        >>> preprocessed = transforms(image=image)
    """
    return Compose(
        [
            Resize(image_size, image_size, always_apply=True),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], always_apply=True),
            ToTensorV2(),
        ]
    )


def get_test_time_augmentation_transforms(image_size: int = 224) -> list:
    """
    Get multiple transform variants for test-time augmentation.

    Args:
        image_size: Target image size

    Returns:
        List of transform compositions for TTA

    Example:
        >>> tta_transforms = get_test_time_augmentation_transforms(224)
        >>> predictions = [model(transform(image=img)['image']) for transform in tta_transforms]
    """
    return [
        # Original
        get_val_transforms(image_size),
        # Horizontal flip
        Compose(
            [
                HorizontalFlip(p=1.0),
                Resize(image_size, image_size, always_apply=True),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], always_apply=True),
                ToTensorV2(),
            ]
        ),
        # Slight brightness adjustment
        Compose(
            [
                RandomBrightness(limit=0.1, p=1.0),
                Resize(image_size, image_size, always_apply=True),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], always_apply=True),
                ToTensorV2(),
            ]
        ),
        # Slight contrast adjustment
        Compose(
            [
                RandomContrast(limit=0.1, p=1.0),
                Resize(image_size, image_size, always_apply=True),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], always_apply=True),
                ToTensorV2(),
            ]
        ),
    ]
