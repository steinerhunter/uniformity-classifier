"""Tests for evaluation utilities."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.evaluate import (
    evaluate_model,
    compare_models,
    plot_confusion_matrix,
    generate_per_image_report,
    plot_sample_images,
)


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


class TestGeneratePerImageReport:
    """Tests for generate_per_image_report function."""

    def test_generates_correct_structure(self):
        """Should generate correct per-image results."""
        paths = [Path("img1.png"), Path("img2.png"), Path("img3.png")]
        true_labels = [0, 1, 1]
        baseline_preds = [0, 0, 1]

        results = generate_per_image_report(paths, true_labels, baseline_preds)

        assert len(results) == 3
        assert results[0]["image"] == "img1.png"
        assert results[0]["ground_truth"] == "PASS"
        assert results[0]["baseline_correct"] == "Yes"
        assert results[1]["baseline_correct"] == "No"

    def test_saves_csv(self):
        """Should save CSV when output_path provided."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "results.csv"
            paths = [Path("img1.png"), Path("img2.png")]
            true_labels = [0, 1]
            baseline_preds = [0, 1]

            generate_per_image_report(
                paths, true_labels, baseline_preds,
                output_path=output_path
            )

            assert output_path.exists()
            content = output_path.read_text()
            assert "img1.png" in content
            assert "PASS" in content

    def test_includes_advanced_predictions(self):
        """Should include advanced model predictions when provided."""
        paths = [Path("img1.png")]
        true_labels = [0]
        baseline_preds = [0]
        advanced_preds = [1]
        advanced_reasoning = ["Found artifact"]

        results = generate_per_image_report(
            paths, true_labels, baseline_preds,
            advanced_predictions=advanced_preds,
            advanced_reasoning=advanced_reasoning
        )

        assert results[0]["advanced_prediction"] == "FAIL"
        assert results[0]["advanced_reasoning"] == "Found artifact"
        assert results[0]["models_agree"] == "No"


class TestPlotSampleImages:
    """Tests for plot_sample_images function."""

    def test_creates_file(self):
        """Should create a sample images visualization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "samples.png"

            images = [
                np.random.randint(0, 255, (64, 64), dtype=np.uint8)
                for _ in range(4)
            ]
            labels = [0, 0, 1, 1]
            paths = [Path(f"img{i}.png") for i in range(4)]

            plot_sample_images(images, labels, paths, output_path, n_samples=2)

            assert output_path.exists()
            assert output_path.stat().st_size > 0
