"""Unit tests for metrics module."""

import pytest
import numpy as np
from deepfake_detector.utils.metrics import (
    eval_state,
    calculate_metrics,
    get_EER_states,
    get_HTER_at_thr,
)


class TestMetrics:
    """Test metrics functions."""

    @pytest.fixture
    def perfect_predictions(self):
        """Perfect predictions for testing."""
        labels = np.array([0, 0, 0, 1, 1, 1])
        probs = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        return probs, labels

    @pytest.fixture
    def random_predictions(self):
        """Random predictions for testing."""
        np.random.seed(42)
        labels = np.random.randint(0, 2, size=100)
        probs = np.random.random(size=100)
        return probs, labels

    def test_eval_state(self, perfect_predictions):
        """Test eval_state function."""
        probs, labels = perfect_predictions
        TN, FN, FP, TP = eval_state(probs, labels, threshold=0.5)

        assert TN == 3  # All fakes correctly identified
        assert TP == 3  # All reals correctly identified
        assert FN == 0  # No false negatives
        assert FP == 0  # No false positives

    def test_calculate_metrics(self, perfect_predictions):
        """Test calculate_metrics function."""
        probs, labels = perfect_predictions
        metrics = calculate_metrics(probs, labels, threshold=0.5)

        assert 'accuracy' in metrics
        assert 'apcer' in metrics
        assert 'npcer' in metrics
        assert 'acer' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics

        # Perfect predictions should have perfect metrics
        assert metrics['accuracy'] == 1.0
        assert metrics['apcer'] == 0.0
        assert metrics['npcer'] == 0.0
        assert metrics['acer'] == 0.0

    def test_get_EER_states(self, random_predictions):
        """Test EER calculation."""
        probs, labels = random_predictions
        EER, optimal_thr, FRR_list, FAR_list = get_EER_states(probs, labels, grid_density=100)

        assert 0.0 <= EER <= 1.0
        assert 0.0 <= optimal_thr <= 1.1
        assert len(FRR_list) > 0
        assert len(FAR_list) > 0
        assert len(FRR_list) == len(FAR_list)

    def test_get_HTER_at_thr(self, random_predictions):
        """Test HTER calculation."""
        probs, labels = random_predictions
        HTER = get_HTER_at_thr(probs, labels, thr=0.5)

        assert 0.0 <= HTER <= 1.0

    def test_metrics_edge_cases(self):
        """Test edge cases for metrics."""
        # All same class
        labels = np.ones(10)
        probs = np.random.random(10)

        metrics = calculate_metrics(probs, labels, threshold=0.5)
        assert 'accuracy' in metrics

        # Empty arrays should raise or handle gracefully
        with pytest.raises((ValueError, ZeroDivisionError, IndexError)):
            calculate_metrics(np.array([]), np.array([]), threshold=0.5)
