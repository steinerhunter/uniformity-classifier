"""
Evaluation utilities for comparing model performance.

Provides:
- Standard classification metrics (accuracy, precision, recall, F1)
- Confusion matrix generation and visualization
- Side-by-side model comparison
- Per-image results report
- Sample image visualization
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# Label constants
LABEL_NAMES = {0: "PASS", 1: "FAIL"}


def evaluate_model(
    true_labels: List[int],
    predictions: List[int],
    model_name: str
) -> Dict[str, float]:
    """
    Compute classification metrics for a model.

    Args:
        true_labels: Ground truth labels (0=PASS, 1=FAIL)
        predictions: Model predictions (0=PASS, 1=FAIL)
        model_name: Name of the model (for logging)

    Returns:
        Dictionary containing:
        - accuracy, precision, recall, f1_score
        - true_negatives, false_positives, false_negatives, true_positives
    """
    metrics: Dict[str, float] = {
        "accuracy": accuracy_score(true_labels, predictions),
        "precision": precision_score(true_labels, predictions, zero_division=0),
        "recall": recall_score(true_labels, predictions, zero_division=0),
        "f1_score": f1_score(true_labels, predictions, zero_division=0),
    }

    # Compute confusion matrix components
    cm = confusion_matrix(true_labels, predictions, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    metrics["true_negatives"] = int(tn)   # Correctly predicted PASS
    metrics["false_positives"] = int(fp)  # PASS incorrectly predicted as FAIL
    metrics["false_negatives"] = int(fn)  # FAIL incorrectly predicted as PASS
    metrics["true_positives"] = int(tp)   # Correctly predicted FAIL

    return metrics


def compare_models(
    baseline_metrics: Dict[str, float],
    advanced_metrics: Dict[str, float]
) -> Dict[str, Dict[str, float]]:
    """
    Compare two models' performance.

    Args:
        baseline_metrics: Metrics from baseline model
        advanced_metrics: Metrics from advanced model

    Returns:
        Dictionary with comparison for each metric:
        {metric: {baseline, advanced, difference, winner}}
    """
    comparison: Dict[str, Dict[str, float]] = {}

    for metric in ["accuracy", "precision", "recall", "f1_score"]:
        baseline_val = baseline_metrics[metric]
        advanced_val = advanced_metrics[metric]
        diff = advanced_val - baseline_val

        comparison[metric] = {
            "baseline": baseline_val,
            "advanced": advanced_val,
            "difference": diff,
            "winner": "advanced" if diff > 0.001 else "baseline" if diff < -0.001 else "tie"
        }

    return comparison


def plot_confusion_matrix(
    true_labels: List[int],
    predictions: List[int],
    model_name: str,
    output_path: Path
) -> None:
    """
    Generate and save a confusion matrix visualization.

    Args:
        true_labels: Ground truth labels
        predictions: Model predictions
        model_name: Name for the title
        output_path: Path to save the figure
    """
    cm = confusion_matrix(true_labels, predictions, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(8, 6))

    # Create heatmap
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    # Labels
    classes = ["PASS", "FAIL"]
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        title=f"Confusion Matrix: {model_name}",
        ylabel="Actual",
        xlabel="Predicted"
    )

    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=20)

    # Add labels for each cell
    cell_labels = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            ax.text(j, i + 0.3, cell_labels[i][j],
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=10, alpha=0.7)

    fig.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def generate_per_image_report(
    paths: List[Path],
    true_labels: List[int],
    baseline_predictions: List[int],
    advanced_predictions: Optional[List[int]] = None,
    advanced_reasoning: Optional[List[str]] = None,
    output_path: Optional[Path] = None
) -> List[Dict]:
    """
    Generate a per-image breakdown of predictions.

    Args:
        paths: List of image file paths
        true_labels: Ground truth labels
        baseline_predictions: Baseline model predictions
        advanced_predictions: Optional advanced model predictions
        advanced_reasoning: Optional reasoning from advanced model
        output_path: Optional path to save CSV

    Returns:
        List of dictionaries with per-image results
    """
    results = []

    for i, path in enumerate(paths):
        true_label = true_labels[i]
        baseline_pred = baseline_predictions[i]
        baseline_correct = true_label == baseline_pred

        row = {
            "image": path.name,
            "ground_truth": LABEL_NAMES[true_label],
            "baseline_prediction": LABEL_NAMES[baseline_pred],
            "baseline_correct": "Yes" if baseline_correct else "No",
        }

        if advanced_predictions:
            advanced_pred = advanced_predictions[i]
            advanced_correct = true_label == advanced_pred
            row["advanced_prediction"] = LABEL_NAMES[advanced_pred]
            row["advanced_correct"] = "Yes" if advanced_correct else "No"

            # Agreement
            row["models_agree"] = "Yes" if baseline_pred == advanced_pred else "No"

        if advanced_reasoning:
            row["advanced_reasoning"] = advanced_reasoning[i]

        results.append(row)

    # Save to CSV if path provided
    if output_path and results:
        fieldnames = list(results[0].keys())
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

    return results


def plot_sample_images(
    images: List[np.ndarray],
    labels: List[int],
    paths: List[Path],
    output_path: Path,
    n_samples: int = 3
) -> None:
    """
    Create a visualization showing sample PASS and FAIL images.

    Args:
        images: List of image arrays
        labels: List of labels
        paths: List of file paths
        output_path: Path to save the figure
        n_samples: Number of samples per class
    """
    # Separate by class
    pass_indices = [i for i, l in enumerate(labels) if l == 0]
    fail_indices = [i for i, l in enumerate(labels) if l == 1]

    # Take up to n_samples from each
    pass_samples = pass_indices[:n_samples]
    fail_samples = fail_indices[:n_samples]

    # Handle case where we don't have enough samples
    n_pass = len(pass_samples)
    n_fail = len(fail_samples)
    n_cols = max(n_pass, n_fail, 1)

    if n_pass == 0 and n_fail == 0:
        # No images to show
        return

    fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 8))

    # Ensure axes is 2D
    if n_cols == 1:
        axes = axes.reshape(2, 1)

    # Plot PASS samples
    for col in range(n_cols):
        ax = axes[0, col]
        if col < n_pass:
            idx = pass_samples[col]
            ax.imshow(images[idx], cmap='gray')
            ax.set_title(f"PASS\n{paths[idx].name}", fontsize=10)
        ax.axis('off')

    # Plot FAIL samples
    for col in range(n_cols):
        ax = axes[1, col]
        if col < n_fail:
            idx = fail_samples[col]
            ax.imshow(images[idx], cmap='gray')
            ax.set_title(f"FAIL\n{paths[idx].name}", fontsize=10)
        ax.axis('off')

    # Add row labels
    fig.text(0.02, 0.75, "PASS\nSamples", ha='left', va='center',
             fontsize=14, fontweight='bold', color='green')
    fig.text(0.02, 0.25, "FAIL\nSamples", ha='left', va='center',
             fontsize=14, fontweight='bold', color='red')

    plt.suptitle("Sample Images from Dataset", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(left=0.1, top=0.9)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
