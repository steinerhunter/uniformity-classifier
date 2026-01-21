"""
Baseline classifier using Random Forest on hand-crafted features.

Random Forest is chosen because:
- Works well with small datasets (hundreds of samples)
- No GPU required, trains in seconds
- Provides feature importance for interpretability
- Robust to different feature scales
- Hard to mess up with default parameters
"""

from typing import List, Tuple
import pickle
from pathlib import Path

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

    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        """
        Initialize classifier.

        Args:
            n_estimators: Number of trees in the forest
            random_state: Random seed for reproducibility
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            class_weight="balanced",  # Handle potential class imbalance
            max_depth=10,  # Prevent overfitting on small datasets
            min_samples_leaf=2,
        )
        self.scaler = StandardScaler()
        self.feature_importances = {}

    def fit(self, features: np.ndarray, labels: List[int], feature_names: List[str] = None):
        """
        Train the classifier.

        Args:
            features: 2D array of shape (n_samples, n_features)
            labels: List of labels (0=PASS, 1=FAIL)
            feature_names: Optional list of feature names for importance tracking
        """
        # Scale features
        features_scaled = self.scaler.fit_transform(features)

        # Train model
        self.model.fit(features_scaled, labels)

        # Store feature importances
        if feature_names:
            importances = self.model.feature_importances_
            self.feature_importances = dict(zip(feature_names, importances))

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

    def get_feature_importances(self) -> dict:
        """Return feature importances sorted by importance."""
        return dict(sorted(
            self.feature_importances.items(),
            key=lambda x: x[1],
            reverse=True
        ))

    def save(self, path: Path):
        """Save model to disk."""
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "scaler": self.scaler}, f)

    def load(self, path: Path):
        """Load model from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.model = data["model"]
            self.scaler = data["scaler"]


# Convenience functions for main.py

def train_baseline(features: np.ndarray, labels: List[int]) -> BaselineClassifier:
    """
    Train a baseline classifier.

    Args:
        features: Training features
        labels: Training labels

    Returns:
        Trained BaselineClassifier
    """
    from src.features import get_feature_names

    classifier = BaselineClassifier()
    classifier.fit(features, labels, feature_names=get_feature_names())

    # Print feature importances
    print("      Feature importances:")
    for name, importance in list(classifier.get_feature_importances().items())[:5]:
        print(f"        {name}: {importance:.3f}")

    return classifier


def predict_baseline(model: BaselineClassifier, features: np.ndarray) -> List[int]:
    """
    Make predictions with trained baseline model.

    Args:
        model: Trained BaselineClassifier
        features: Test features

    Returns:
        List of predicted labels
    """
    return model.predict(features)
