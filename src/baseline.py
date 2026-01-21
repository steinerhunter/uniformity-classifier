"""
Baseline classifier using Random Forest on hand-crafted features.

Random Forest is chosen because:
- Works well with small datasets (hundreds of samples)
- No GPU required, trains in seconds
- Provides feature importance for interpretability
- Robust to different feature scales
- Hard to mess up with default parameters
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


class BaselineClassifier:
    """
    Random Forest classifier with feature scaling.

    Attributes:
        model: Trained RandomForestClassifier
        scaler: StandardScaler for feature normalization
        feature_importances: Dict of feature name -> importance score
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 10,
        random_state: int = 42
    ) -> None:
        """
        Initialize classifier.

        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of trees (prevents overfitting)
            random_state: Random seed for reproducibility
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            class_weight="balanced",  # Handle potential class imbalance
            max_depth=max_depth,
            min_samples_leaf=2,
        )
        self.scaler = StandardScaler()
        self.feature_importances: Dict[str, float] = {}

    def fit(
        self,
        features: np.ndarray,
        labels: List[int],
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Train the classifier.

        Args:
            features: 2D array of shape (n_samples, n_features)
            labels: List of labels (0=PASS, 1=FAIL)
            feature_names: Optional list of feature names for importance tracking

        Returns:
            Dictionary of feature name -> importance (sorted by importance)
        """
        # Scale features
        features_scaled = self.scaler.fit_transform(features)

        # Train model
        self.model.fit(features_scaled, labels)

        # Store feature importances
        if feature_names:
            importances = self.model.feature_importances_
            self.feature_importances = dict(zip(feature_names, importances))

        return self.get_feature_importances()

    def predict(self, features: np.ndarray) -> List[int]:
        """
        Predict labels for new samples.

        Args:
            features: 2D array of shape (n_samples, n_features)

        Returns:
            List of predicted labels (0=PASS, 1=FAIL)
        """
        features_scaled = self.scaler.transform(features)
        return self.model.predict(features_scaled).tolist()

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            features: 2D array of shape (n_samples, n_features)

        Returns:
            2D array of shape (n_samples, 2) with [P(PASS), P(FAIL)]
        """
        features_scaled = self.scaler.transform(features)
        return self.model.predict_proba(features_scaled)

    def get_feature_importances(self) -> Dict[str, float]:
        """Return feature importances sorted by importance (descending)."""
        return dict(sorted(
            self.feature_importances.items(),
            key=lambda x: x[1],
            reverse=True
        ))

    def save(self, path: Path) -> None:
        """Save model to disk."""
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "scaler": self.scaler,
                "feature_importances": self.feature_importances
            }, f)

    @classmethod
    def load(cls, path: Path) -> "BaselineClassifier":
        """Load model from disk."""
        instance = cls()
        with open(path, "rb") as f:
            data = pickle.load(f)
            instance.model = data["model"]
            instance.scaler = data["scaler"]
            instance.feature_importances = data.get("feature_importances", {})
        return instance


# ---------------------------------------------------------------------------
# Convenience functions for main.py
# ---------------------------------------------------------------------------

def train_baseline(
    features: np.ndarray,
    labels: List[int],
    feature_names: Optional[List[str]] = None
) -> Tuple[BaselineClassifier, Dict[str, float]]:
    """
    Train a baseline classifier.

    Args:
        features: Training features (n_samples, n_features)
        labels: Training labels (0=PASS, 1=FAIL)
        feature_names: Optional feature names for importance tracking

    Returns:
        Tuple of (trained classifier, feature importances dict)
    """
    classifier = BaselineClassifier()
    importances = classifier.fit(features, labels, feature_names=feature_names)
    return classifier, importances


def predict_baseline(model: BaselineClassifier, features: np.ndarray) -> List[int]:
    """
    Make predictions with trained baseline model.

    Args:
        model: Trained BaselineClassifier
        features: Test features (n_samples, n_features)

    Returns:
        List of predicted labels (0=PASS, 1=FAIL)
    """
    return model.predict(features)
