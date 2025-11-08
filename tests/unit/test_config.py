"""Unit tests for config module."""

import pytest
from pathlib import Path
from deepfake_detector.config import Config


class TestConfig:
    """Test Config class."""

    def test_default_config(self):
        """Test default configuration."""
        config = Config()

        assert config.model_name == 'efficientnet-b1'
        assert config.batch_size == 32
        assert config.num_epochs == 20
        assert config.learning_rate == 8e-4

    def test_custom_config(self):
        """Test custom configuration."""
        config = Config(
            model_name='efficientnet-b0',
            batch_size=64,
            num_epochs=10,
        )

        assert config.model_name == 'efficientnet-b0'
        assert config.batch_size == 64
        assert config.num_epochs == 10

    def test_config_validation(self):
        """Test configuration validation."""
        # Invalid batch size
        with pytest.raises(ValueError):
            Config(batch_size=0)

        # Invalid epochs
        with pytest.raises(ValueError):
            Config(num_epochs=-1)

        # Invalid learning rate
        with pytest.raises(ValueError):
            Config(learning_rate=2.0)

    def test_config_to_dict(self):
        """Test converting config to dictionary."""
        config = Config(model_name='efficientnet-b0')
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert 'model_name' in config_dict
        assert config_dict['model_name'] == 'efficientnet-b0'

    def test_config_from_dict(self):
        """Test creating config from dictionary."""
        config_dict = {
            'model_name': 'efficientnet-b2',
            'batch_size': 16,
        }

        config = Config.from_dict(config_dict)

        assert config.model_name == 'efficientnet-b2'
        assert config.batch_size == 16

    def test_config_save_load_yaml(self, tmp_path):
        """Test saving and loading YAML config."""
        config = Config(model_name='efficientnet-b3', batch_size=128)

        # Save
        config_path = tmp_path / 'config.yaml'
        config.save(config_path)

        assert config_path.exists()

        # Load
        loaded_config = Config.load(config_path)

        assert loaded_config.model_name == config.model_name
        assert loaded_config.batch_size == config.batch_size

    def test_config_save_load_json(self, tmp_path):
        """Test saving and loading JSON config."""
        config = Config(model_name='efficientnet-b4', batch_size=256)

        # Save
        config_path = tmp_path / 'config.json'
        config.save(config_path)

        assert config_path.exists()

        # Load
        loaded_config = Config.load(config_path)

        assert loaded_config.model_name == config.model_name
        assert loaded_config.batch_size == config.batch_size

    def test_config_invalid_file_format(self, tmp_path):
        """Test invalid file format raises error."""
        config = Config()
        config_path = tmp_path / 'config.txt'

        with pytest.raises(ValueError):
            config.save(config_path)
