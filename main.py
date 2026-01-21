"""
Uniformity Test Pass/Fail Classifier
=====================================

Entry point for running the full classification pipeline.

Usage:
    python main.py

This script:
1. Loads and splits the dataset
2. Trains and evaluates the baseline model (Random Forest)
3. Runs and evaluates the advanced model (GPT-4o Vision)
4. Outputs comparison metrics and confusion matrices
"""

import json
from pathlib import Path

from src.data_loader import load_dataset, split_dataset
from src.features import extract_features
from src.baseline import train_baseline, predict_baseline
from src.advanced import predict_advanced
from src.evaluate import evaluate_model, compare_models, plot_confusion_matrix


def main():
    print("=" * 60)
    print("Uniformity Test Pass/Fail Classifier")
    print("=" * 60)

    # Paths
    data_dir = Path("data")
    output_dir = Path("outputs")
    cache_dir = Path("cache")

    output_dir.mkdir(exist_ok=True)
    cache_dir.mkdir(exist_ok=True)

    # Step 1: Load and split data
    print("\n[1/5] Loading dataset...")
    images, labels, paths = load_dataset(data_dir)
    print(f"      Loaded {len(images)} images ({sum(labels)} FAIL, {len(labels) - sum(labels)} PASS)")

    train_data, test_data = split_dataset(images, labels, paths, test_size=0.2, random_state=42)
    train_images, train_labels, _ = train_data
    test_images, test_labels, test_paths = test_data
    print(f"      Train: {len(train_images)}, Test: {len(test_images)}")

    # Step 2: Extract features for baseline
    print("\n[2/5] Extracting features for baseline model...")
    train_features = extract_features(train_images)
    test_features = extract_features(test_images)
    print(f"      Extracted {train_features.shape[1]} features per image")

    # Step 3: Train and evaluate baseline
    print("\n[3/5] Training baseline model (Random Forest)...")
    model = train_baseline(train_features, train_labels)
    baseline_predictions = predict_baseline(model, test_features)
    baseline_metrics = evaluate_model(test_labels, baseline_predictions, "Baseline (Random Forest)")

    # Step 4: Run and evaluate advanced model
    print("\n[4/5] Running advanced model (GPT-4o Vision)...")
    advanced_predictions, advanced_reasoning = predict_advanced(
        test_images,
        test_paths,
        cache_dir=cache_dir
    )
    advanced_metrics = evaluate_model(test_labels, advanced_predictions, "Advanced (GPT-4o)")

    # Step 5: Compare and output results
    print("\n[5/5] Generating comparison and outputs...")
    comparison = compare_models(baseline_metrics, advanced_metrics)

    # Save confusion matrices
    plot_confusion_matrix(
        test_labels,
        baseline_predictions,
        "Baseline (Random Forest)",
        output_dir / "confusion_matrix_baseline.png"
    )
    plot_confusion_matrix(
        test_labels,
        advanced_predictions,
        "Advanced (GPT-4o)",
        output_dir / "confusion_matrix_advanced.png"
    )

    # Save full results
    results = {
        "baseline": baseline_metrics,
        "advanced": advanced_metrics,
        "comparison": comparison,
        "sample_reasoning": advanced_reasoning[:3] if advanced_reasoning else []
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"\n{'Metric':<20} {'Baseline':<15} {'GPT-4o':<15}")
    print("-" * 50)
    for metric in ["accuracy", "precision", "recall", "f1_score"]:
        b_val = baseline_metrics[metric]
        a_val = advanced_metrics[metric]
        print(f"{metric:<20} {b_val:<15.3f} {a_val:<15.3f}")

    print(f"\nOutputs saved to: {output_dir.absolute()}")
    print("Done!")


if __name__ == "__main__":
    main()
