#!/usr/bin/env python3
"""
Uniformity Test Pass/Fail Classifier
=====================================

Entry point for running the full classification pipeline.

Usage:
    python main.py                          # Run with defaults
    python main.py --data-dir ./my_data     # Custom data directory
    python main.py --test-size 0.3          # 30% test split
    python main.py --no-advanced            # Skip GPT-4o (baseline only)
    python main.py --verbose                # Debug logging

This script:
1. Loads and splits the dataset
2. Trains and evaluates the baseline model (Random Forest)
3. Runs and evaluates the advanced model (GPT-4o Vision)
4. Outputs comparison metrics, confusion matrices, and per-image results
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

from src.data_loader import load_dataset, split_dataset
from src.features import extract_features, get_feature_names
from src.baseline import train_baseline, predict_baseline
from src.advanced import predict_advanced
from src.evaluate import (
    evaluate_model,
    compare_models,
    plot_confusion_matrix,
    generate_per_image_report,
    plot_sample_images,
)


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Configure logging for the application."""
    level = logging.DEBUG if verbose else logging.INFO

    # Create formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S"
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # Configure root logger
    logger = logging.getLogger("uniformity")
    logger.setLevel(level)
    logger.addHandler(console_handler)

    return logger


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Uniformity Test Pass/Fail Classifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                              Run with default settings
  python main.py --data-dir ./custom_data     Use custom data directory
  python main.py --test-size 0.3              Use 30% of data for testing
  python main.py --no-advanced                Skip GPT-4o model (faster)
  python main.py --verbose                    Enable debug logging
        """
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing PASS/ and FAIL/ subdirectories (default: ./data)"
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory for output files (default: ./outputs)"
    )

    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("cache"),
        help="Directory for API response cache (default: ./cache)"
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data for testing (default: 0.2)"
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )

    parser.add_argument(
        "--no-advanced",
        action="store_true",
        help="Skip the advanced (GPT-4o) model"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose (debug) logging"
    )

    return parser.parse_args()


def main() -> int:
    """
    Main entry point.

    Returns:
        Exit code (0 for success, 1 for error)
    """
    args = parse_args()
    logger = setup_logging(args.verbose)

    logger.info("=" * 60)
    logger.info("Uniformity Test Pass/Fail Classifier")
    logger.info("=" * 60)

    # Ensure directories exist
    args.output_dir.mkdir(exist_ok=True)
    args.cache_dir.mkdir(exist_ok=True)

    # -------------------------------------------------------------------------
    # Step 1: Load and split data
    # -------------------------------------------------------------------------
    logger.info("[1/6] Loading dataset from %s", args.data_dir)

    try:
        images, labels, paths = load_dataset(args.data_dir)
    except FileNotFoundError as e:
        logger.error("Dataset not found: %s", e)
        logger.error("Please ensure data is in %s/PASS and %s/FAIL",
                     args.data_dir, args.data_dir)
        return 1

    pass_count = len(labels) - sum(labels)
    fail_count = sum(labels)
    logger.info("Loaded %d images (%d PASS, %d FAIL)", len(images), pass_count, fail_count)

    train_data, test_data = split_dataset(
        images, labels, paths,
        test_size=args.test_size,
        random_state=args.random_state
    )
    train_images, train_labels, train_paths = train_data
    test_images, test_labels, test_paths = test_data

    logger.info("Split: %d training, %d testing", len(train_images), len(test_images))

    # -------------------------------------------------------------------------
    # Step 2: Visualize sample images
    # -------------------------------------------------------------------------
    logger.info("[2/6] Generating sample image visualization")

    plot_sample_images(
        images, labels, paths,
        output_path=args.output_dir / "sample_images.png",
        n_samples=3
    )

    # -------------------------------------------------------------------------
    # Step 3: Extract features and train baseline
    # -------------------------------------------------------------------------
    logger.info("[3/6] Training baseline model (Random Forest)")

    logger.debug("Extracting features from training images...")
    train_features = extract_features(train_images)
    test_features = extract_features(test_images)
    logger.info("Extracted %d features per image", train_features.shape[1])

    model, feature_importances = train_baseline(
        train_features,
        train_labels,
        feature_names=get_feature_names()
    )

    logger.info("Top features: %s",
                ", ".join(f"{k}={v:.3f}" for k, v in list(feature_importances.items())[:3]))

    baseline_predictions = predict_baseline(model, test_features)
    baseline_metrics = evaluate_model(test_labels, baseline_predictions, "Baseline")

    logger.info("Baseline accuracy: %.1f%%", baseline_metrics["accuracy"] * 100)

    # -------------------------------------------------------------------------
    # Step 4: Run advanced model (optional)
    # -------------------------------------------------------------------------
    advanced_predictions: Optional[list] = None
    advanced_metrics: Optional[dict] = None
    advanced_reasoning: Optional[list] = None

    if not args.no_advanced:
        logger.info("[4/6] Running advanced model (GPT-4o Vision)")

        advanced_predictions, advanced_reasoning = predict_advanced(
            test_images,
            test_paths,
            cache_dir=args.cache_dir,
            logger=logger
        )
        advanced_metrics = evaluate_model(test_labels, advanced_predictions, "GPT-4o")

        logger.info("GPT-4o accuracy: %.1f%%", advanced_metrics["accuracy"] * 100)
    else:
        logger.info("[4/6] Skipping advanced model (--no-advanced flag)")

    # -------------------------------------------------------------------------
    # Step 5: Generate outputs
    # -------------------------------------------------------------------------
    logger.info("[5/6] Generating outputs")

    # Confusion matrices
    plot_confusion_matrix(
        test_labels,
        baseline_predictions,
        "Baseline (Random Forest)",
        args.output_dir / "confusion_matrix_baseline.png"
    )
    logger.debug("Saved baseline confusion matrix")

    if advanced_predictions:
        plot_confusion_matrix(
            test_labels,
            advanced_predictions,
            "Advanced (GPT-4o)",
            args.output_dir / "confusion_matrix_advanced.png"
        )
        logger.debug("Saved advanced confusion matrix")

    # Per-image results
    per_image_results = generate_per_image_report(
        test_paths,
        test_labels,
        baseline_predictions,
        advanced_predictions,
        advanced_reasoning,
        output_path=args.output_dir / "per_image_results.csv"
    )
    logger.info("Saved per-image results to per_image_results.csv")

    # -------------------------------------------------------------------------
    # Step 6: Compare and summarize
    # -------------------------------------------------------------------------
    logger.info("[6/6] Generating summary")

    comparison = None
    if advanced_metrics:
        comparison = compare_models(baseline_metrics, advanced_metrics)

    # Build results object
    results = {
        "dataset": {
            "total_images": len(images),
            "pass_count": pass_count,
            "fail_count": fail_count,
            "test_size": len(test_images),
            "train_size": len(train_images),
        },
        "baseline": {
            "metrics": baseline_metrics,
            "feature_importances": feature_importances,
        },
        "advanced": {
            "metrics": advanced_metrics,
            "sample_reasoning": advanced_reasoning[:5] if advanced_reasoning else None,
        } if advanced_metrics else None,
        "comparison": comparison,
    }

    # Save JSON results
    with open(args.output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.debug("Saved results.json")

    # Print summary table
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    if advanced_metrics:
        print(f"\n{'Metric':<20} {'Baseline':<15} {'GPT-4o':<15} {'Winner':<10}")
        print("-" * 60)
        for metric in ["accuracy", "precision", "recall", "f1_score"]:
            b_val = baseline_metrics[metric]
            a_val = advanced_metrics[metric]
            winner = comparison[metric]["winner"] if comparison else "-"
            print(f"{metric:<20} {b_val:<15.3f} {a_val:<15.3f} {winner:<10}")
    else:
        print(f"\n{'Metric':<20} {'Baseline':<15}")
        print("-" * 35)
        for metric in ["accuracy", "precision", "recall", "f1_score"]:
            print(f"{metric:<20} {baseline_metrics[metric]:<15.3f}")

    print(f"\nOutputs saved to: {args.output_dir.absolute()}")
    print("=" * 60)

    logger.info("Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
