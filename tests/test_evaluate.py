"""Tests for evaluation utilities."""

import tempfile
from pathlib import Path

import pytest

from src.evaluate import evaluate_model, compare_models, plot_confusion_matrix


class TestEvaluateModel:
    """Tests for evaluate_model function."""

    def test_perfect_predictions(self):
        """Perfect predictions should have all metrics = 1.0."""
        true_labels = [0, 0, 1, 1, 1]
        predictions = [0, 0, 1, 1, 1]

        metrics = evaluate_model(true_labels, predictions, "Test Model")

        assert metrics["accuracy"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1_score"] == 1.0

    def test_all_wrong_predictions(self):
        """All wrong predictions should have accuracy = 0."""
        true_labels = [0, 0, 1, 1]
        predictions = [1, 1, 0, 0]

        metrics = evaluate_model(true_labels, predictions, "Test Model")

        assert metrics["accuracy"] == 0.0

    def test_confusion_matrix_components(self):
        """Should correctly compute TN, FP, FN, TP."""
        # 2 true PASS, 2 true FAIL
        # Predict: correct PASS, wrong PASS->FAIL, wrong FAIL->PASS, correct FAIL
        true_labels = [0, 0, 1, 1]
        predictions = [0, 1, 0, 1]

        metrics = evaluate_model(true_labels, predictions, "Test Model")

        assert metrics["true_negatives"] == 1   # Correct PASS
        assert metrics["false_positives"] == 1  # PASS predicted as FAIL
        assert metrics["false_negatives"] == 1  # FAIL predicted as PASS
        assert metrics["true_positives"] == 1   # Correct FAIL

    def test_handles_all_pass(self):
        """Should handle case where all labels are PASS."""
        true_labels = [0, 0, 0]
        predictions = [0, 0, 0]

        metrics = evaluate_model(true_labels, predictions, "Test Model")

        assert metrics["accuracy"] == 1.0
        assert metrics["true_negatives"] == 3


class TestCompareModels:
    """Tests for compare_models function."""

    def test_identifies_winner(self):
        """Should correctly identify which model is better."""
        baseline = {"accuracy": 0.8, "precision": 0.7, "recall": 0.9, "f1_score": 0.78}
        advanced = {"accuracy": 0.9, "precision": 0.85, "recall": 0.85, "f1_score": 0.85}

        comparison = compare_models(baseline, advanced)

        assert comparison["accuracy"]["winner"] == "advanced"
        assert comparison["f1_score"]["winner"] == "advanced"

    def test_identifies_tie(self):
        """Should identify ties."""
        baseline = {"accuracy": 0.8, "precision": 0.8, "recall": 0.8, "f1_score": 0.8}
        advanced = {"accuracy": 0.8, "precision": 0.8, "recall": 0.8, "f1_score": 0.8}

        comparison = compare_models(baseline, advanced)

        assert comparison["accuracy"]["winner"] == "tie"

    def test_computes_difference(self):
        """Should compute correct differences."""
        baseline = {"accuracy": 0.8, "precision": 0.7, "recall": 0.9, "f1_score": 0.78}
        advanced = {"accuracy": 0.9, "precision": 0.85, "recall": 0.85, "f1_score": 0.85}

        comparison = compare_models(baseline, advanced)

        assert abs(comparison["accuracy"]["difference"] - 0.1) < 0.001


class TestPlotConfusionMatrix:
    """Tests for plot_confusion_matrix function."""

    def test_creates_file(self):
        """Should create an image file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_cm.png"

            true_labels = [0, 0, 1, 1]
            predictions = [0, 1, 0, 1]

            plot_confusion_matrix(true_labels, predictions, "Test", output_path)

            assert output_path.exists()
            assert output_path.stat().st_size > 0
