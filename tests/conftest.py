"""Pytest configuration and fixtures."""

import numpy as np
import pytest
import torch


@pytest.fixture
def device():
    """Get device for testing."""
    return torch.device("cpu")


@pytest.fixture
def sample_image():
    """Create a sample RGB image."""
    return np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)


@pytest.fixture
def sample_tensor():
    """Create a sample image tensor."""
    return torch.randn(1, 3, 224, 224)


@pytest.fixture
def temp_checkpoint(tmp_path):
    """Create a temporary checkpoint path."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    return checkpoint_dir / "test_model.pth"


@pytest.fixture
def mock_config():
    """Create a mock configuration."""
    from deepfake_detector.config import Config

    return Config(
        model_name="efficientnet-b0",
        batch_size=2,
        num_epochs=1,
        learning_rate=1e-4,
    )
