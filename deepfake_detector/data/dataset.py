"""
Dataset class for DeepFake Detection
Handles loading and preprocessing of facial images.
"""

import os
import glob
from typing import List, Tuple, Optional, Callable
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from albumentations import Compose
import logging

logger = logging.getLogger(__name__)


class DeepFakeDataset(Dataset):
    """
    PyTorch Dataset for DeepFake Detection.

    Supports multiple data directories and balanced sampling.

    Args:
        data_config: List of tuples [(directory_path, num_samples), ...]
        is_real: Whether this dataset contains real (True) or fake (False) images
        transform: Albumentations transform composition
        image_extensions: Tuple of supported image extensions

    Example:
        >>> config = [("/path/to/real", 1000), ("/path/to/real2", 500)]
        >>> dataset = DeepFakeDataset(config, is_real=True, transform=transforms)
        >>> image, label = dataset[0]
    """

    def __init__(
        self,
        data_config: List[Tuple[str, int]],
        is_real: bool = True,
        transform: Optional[Compose] = None,
        image_extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png"),
    ):
        super().__init__()

        self.data_config = data_config
        self.is_real = is_real
        self.transform = transform
        self.image_extensions = image_extensions

        # Build dataset
        self.data = self._build_dataset()

        logger.info(
            f"{'Real' if is_real else 'Fake'} dataset initialized: " f"{len(self.data)} samples"
        )

    def _build_dataset(self) -> pd.DataFrame:
        """Build dataset from configuration."""
        all_data = []

        for directory, sample_num in self.data_config:
            if not os.path.exists(directory):
                logger.warning(f"Directory not found: {directory}, skipping...")
                continue

            # Collect all image paths
            image_paths = []
            for ext in self.image_extensions:
                image_paths.extend(glob.glob(os.path.join(directory, f"*{ext}")))

            if len(image_paths) == 0:
                logger.warning(f"No images found in {directory}")
                continue

            # Create dataframe
            df = pd.DataFrame(image_paths, columns=["image_path"])
            df["real"] = 1.0 if self.is_real else 0.0
            df["fake"] = 0.0 if self.is_real else 1.0

            # Sample if necessary
            if sample_num > 0 and len(df) >= sample_num:
                df = df.sample(n=sample_num, random_state=42, replace=False)
            elif sample_num > len(df):
                logger.warning(
                    f"Requested {sample_num} samples but only {len(df)} available in {directory}"
                )

            all_data.append(df)

            logger.debug(f"Loaded {len(df)} samples from {directory}")

        if not all_data:
            raise ValueError("No valid data found in any directory")

        return pd.concat(all_data, ignore_index=True)

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get item by index.

        Args:
            idx: Index of the sample

        Returns:
            Tuple of (image_tensor, label_tensor)
        """
        # Get image path and labels
        image_path = self.data.loc[idx, "image_path"]
        real_label = self.data.loc[idx, "real"]
        fake_label = self.data.loc[idx, "fake"]

        # Load image
        image_bgr = cv2.imread(image_path)

        if image_bgr is None:
            logger.error(f"Failed to load image: {image_path}")
            # Return a black image as fallback
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            # Convert BGR to RGB
            image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]

        # Create label (class index)
        label = torch.tensor(int(fake_label), dtype=torch.long)

        return image, label

    def get_labels(self) -> np.ndarray:
        """Get all labels as numpy array."""
        return self.data["fake"].values.astype(int)

    def get_class_weights(self) -> torch.Tensor:
        """
        Calculate class weights for balanced training.

        Returns:
            Tensor of class weights
        """
        labels = self.get_labels()
        class_counts = np.bincount(labels)
        total = len(labels)
        weights = total / (len(class_counts) * class_counts)
        return torch.tensor(weights, dtype=torch.float32)


def create_combined_dataset(
    real_config: List[Tuple[str, int]],
    fake_config: List[Tuple[str, int]],
    transform: Optional[Compose] = None,
) -> Dataset:
    """
    Create a combined dataset with both real and fake samples.

    Args:
        real_config: Configuration for real images
        fake_config: Configuration for fake images
        transform: Transform to apply

    Returns:
        Combined dataset

    Example:
        >>> real_cfg = [("/path/real", 1000)]
        >>> fake_cfg = [("/path/fake", 1000)]
        >>> dataset = create_combined_dataset(real_cfg, fake_cfg, transforms)
    """
    real_dataset = DeepFakeDataset(real_config, is_real=True, transform=transform)
    fake_dataset = DeepFakeDataset(fake_config, is_real=False, transform=transform)

    # Combine datasets
    combined_data = pd.concat([real_dataset.data, fake_dataset.data], ignore_index=True)

    # Create new dataset with combined data
    class CombinedDataset(Dataset):
        def __init__(self, data, transform):
            self.data = data
            self.transform = transform

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            image_path = self.data.loc[idx, "image_path"]
            fake_label = self.data.loc[idx, "fake"]

            image_bgr = cv2.imread(image_path)
            if image_bgr is None:
                image = np.zeros((224, 224, 3), dtype=np.uint8)
            else:
                image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

            if self.transform is not None:
                transformed = self.transform(image=image)
                image = transformed["image"]

            label = torch.tensor(int(fake_label), dtype=torch.long)
            return image, label

    return CombinedDataset(combined_data, transform)
