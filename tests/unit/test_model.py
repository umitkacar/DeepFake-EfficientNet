"""Unit tests for model module."""

import pytest
import torch
from deepfake_detector.models import DeepFakeDetector


class TestDeepFakeDetector:
    """Test DeepFakeDetector class."""

    def test_model_initialization(self):
        """Test model can be initialized."""
        model = DeepFakeDetector(model_name='efficientnet-b0', pretrained=False)
        assert model is not None
        assert model.model_name == 'efficientnet-b0'
        assert model.num_classes == 2

    def test_model_forward_pass(self, sample_tensor, device):
        """Test forward pass produces correct output shape."""
        model = DeepFakeDetector(model_name='efficientnet-b0', pretrained=False)
        model = model.to(device)
        model.eval()

        with torch.no_grad():
            output = model(sample_tensor)

        assert output.shape == (1, 2)
        assert not torch.isnan(output).any()

    def test_get_image_size(self):
        """Test getting correct image size for model."""
        model = DeepFakeDetector(model_name='efficientnet-b0')
        image_size = model.get_image_size()
        assert image_size == 224

    def test_parameter_counting(self):
        """Test parameter counting."""
        model = DeepFakeDetector(model_name='efficientnet-b0', pretrained=False)
        total_params, trainable_params = model.count_parameters()

        assert total_params > 0
        assert trainable_params > 0
        assert trainable_params <= total_params

    def test_freeze_backbone(self):
        """Test freezing backbone."""
        model = DeepFakeDetector(model_name='efficientnet-b0', pretrained=False)

        # Freeze backbone
        model.freeze_backbone(freeze=True)

        # Check backbone is frozen (except classification head)
        for name, param in model.named_parameters():
            if '_fc' not in name:
                assert not param.requires_grad
            else:
                assert param.requires_grad

    def test_checkpoint_save_load(self, temp_checkpoint, device):
        """Test saving and loading checkpoints."""
        model = DeepFakeDetector(model_name='efficientnet-b0', pretrained=False)

        # Save checkpoint
        model.save_checkpoint(str(temp_checkpoint), epoch=5)

        # Load checkpoint
        new_model = DeepFakeDetector(model_name='efficientnet-b0', pretrained=False)
        new_model.load_checkpoint(str(temp_checkpoint), device=str(device))

        # Check models have same parameters
        for (name1, param1), (name2, param2) in zip(
            model.named_parameters(), new_model.named_parameters()
        ):
            assert name1 == name2
            assert torch.allclose(param1, param2)
