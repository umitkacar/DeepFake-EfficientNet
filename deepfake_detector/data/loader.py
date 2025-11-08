"""
DataLoader creation and configuration.
Handles multi-process data loading with proper worker initialization.
"""

import torch
from torch.utils.data import DataLoader, Dataset
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def create_dataloaders(
    train_dataset: Optional[Dataset] = None,
    val_dataset: Optional[Dataset] = None,
    test_dataset: Optional[Dataset] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last_train: bool = True
) -> Tuple[Optional[DataLoader], ...]:
    """
    Create PyTorch DataLoaders for train/val/test datasets.

    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        batch_size: Batch size for all loaders
        num_workers: Number of workers for data loading
        pin_memory: Whether to pin memory (faster transfer to GPU)
        drop_last_train: Whether to drop last incomplete batch in training

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
        Returns None for datasets that are not provided

    Example:
        >>> train_loader, val_loader, _ = create_dataloaders(
        ...     train_dataset=train_ds,
        ...     val_dataset=val_ds,
        ...     batch_size=64,
        ...     num_workers=8
        ... )
    """
    train_loader = None
    val_loader = None
    test_loader = None

    if train_dataset is not None:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last_train,
            persistent_workers=num_workers > 0
        )
        logger.info(
            f"Train DataLoader created: {len(train_dataset)} samples, "
            f"{len(train_loader)} batches"
        )

    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
            persistent_workers=num_workers > 0
        )
        logger.info(
            f"Validation DataLoader created: {len(val_dataset)} samples, "
            f"{len(val_loader)} batches"
        )

    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
            persistent_workers=num_workers > 0
        )
        logger.info(
            f"Test DataLoader created: {len(test_dataset)} samples, "
            f"{len(test_loader)} batches"
        )

    return train_loader, val_loader, test_loader


def get_optimal_num_workers() -> int:
    """
    Automatically determine optimal number of workers.

    Returns:
        Recommended number of workers

    Example:
        >>> num_workers = get_optimal_num_workers()
        >>> print(f"Using {num_workers} workers")
    """
    import os
    import multiprocessing as mp

    # Get number of CPUs
    num_cpus = mp.cpu_count()

    # Heuristic: use 75% of CPUs, but not more than 8
    optimal = min(max(1, int(num_cpus * 0.75)), 8)

    logger.debug(f"Detected {num_cpus} CPUs, recommending {optimal} workers")
    return optimal
