"""
Evaluation utilities for comparing model performance.

Provides:
- Standard classification metrics (accuracy, precision, recall, F1)
- Confusion matrix generation and visualization
- Side-by-side model comparison
"""

from pathlib import Path
from typing import List, Dict

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


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
        model_name: Name of the model (for display)

    Returns:
        Dictionary of metric name -> value
    """
    metrics = {
        "accuracy": accuracy_score(true_labels, predictions),
        "precision": precision_score(true_labels, predictions, zero_division=0),
        "recall": recall_score(true_labels, predictions, zero_division=0),
        "f1_score": f1_score(true_labels, predictions, zero_division=0),
    }

    # Compute confusion matrix components
    cm = confusion_matrix(true_labels, predictions, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    metrics["true_negatives"] = int(tn)  # Correctly predicted PASS
    metrics["false_positives"] = int(fp)  # PASS incorrectly predicted as FAIL
    metrics["false_negatives"] = int(fn)  # FAIL incorrectly predicted as PASS
    metrics["true_positives"] = int(tp)  # Correctly predicted FAIL

    # Print summary
    print(f"\n      {model_name} Results:")
    print(f"        Accuracy:  {metrics['accuracy']:.3f}")
    print(f"        Precision: {metrics['precision']:.3f}")
    print(f"        Recall:    {metrics['recall']:.3f}")
    print(f"        F1 Score:  {metrics['f1_score']:.3f}")
    print(f"        Confusion: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

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
        Dictionary with comparison results
    """
    comparison = {}

    for metric in ["accuracy", "precision", "recall", "f1_score"]:
        baseline_val = baseline_metrics[metric]
        advanced_val = advanced_metrics[metric]
        diff = advanced_val - baseline_val

        comparison[metric] = {
            "baseline": baseline_val,
            "advanced": advanced_val,
            "difference": diff,
            "winner": "advanced" if diff > 0 else "baseline" if diff < 0 else "tie"
        }

    return comparison


def plot_confusion_matrix(
    true_labels: List[int],
    predictions: List[int],
    model_name: str,
    output_path: Path
):
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

    print(f"      Saved confusion matrix to {output_path}")
