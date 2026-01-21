"""Tests for feature extraction utilities."""

import numpy as np
import pytest

from src.features import (
    compute_mean_intensity,
    compute_std_intensity,
    compute_coef_of_variation,
    compute_gradient_magnitude,
    compute_max_local_variance,
    compute_histogram_entropy,
    compute_percentile_range,
    compute_center_vs_edge_ratio,
    extract_features,
    get_feature_names,
)


class TestIndividualFeatures:
    """Tests for individual feature functions."""

    def test_mean_intensity_uniform_image(self):
        """Mean of uniform image should equal pixel value."""
        img = np.full((64, 64), 128, dtype=np.uint8)
        assert compute_mean_intensity(img) == 128.0

    def test_std_intensity_uniform_image(self):
        """Std of uniform image should be zero."""
        img = np.full((64, 64), 128, dtype=np.uint8)
        assert compute_std_intensity(img) == 0.0

    def test_std_intensity_varied_image(self):
        """Std of varied image should be positive."""
        img = np.array([[0, 255], [255, 0]], dtype=np.uint8)
        assert compute_std_intensity(img) > 0

    def test_coef_of_variation_uniform(self):
        """CV of uniform image should be zero."""
        img = np.full((64, 64), 128, dtype=np.uint8)
        assert compute_coef_of_variation(img) == 0.0

    def test_coef_of_variation_handles_zero_mean(self):
        """CV should handle zero mean gracefully."""
        img = np.zeros((64, 64), dtype=np.uint8)
        assert compute_coef_of_variation(img) == 0.0

    def test_gradient_magnitude_uniform_low(self):
        """Uniform image should have low gradient magnitude."""
        uniform = np.full((64, 64), 128, dtype=np.uint8)
        assert compute_gradient_magnitude(uniform) < 1.0

    def test_gradient_magnitude_edge_high(self):
        """Image with sharp edge should have higher gradient."""
        # Left half black, right half white
        edge = np.zeros((64, 64), dtype=np.uint8)
        edge[:, 32:] = 255
        assert compute_gradient_magnitude(edge) > 10.0

    def test_max_local_variance_uniform_low(self):
        """Uniform image should have low max local variance."""
        img = np.full((64, 64), 128, dtype=np.uint8)
        assert compute_max_local_variance(img) == 0.0

    def test_max_local_variance_detects_hotspot(self):
        """Should detect high variance region."""
        img = np.full((64, 64), 128, dtype=np.uint8)
        # Add a high-contrast region
        img[20:30, 20:30] = np.random.randint(0, 255, (10, 10), dtype=np.uint8)
        assert compute_max_local_variance(img) > 0

    def test_histogram_entropy_uniform_low(self):
        """Uniform image should have low entropy."""
        uniform = np.full((64, 64), 128, dtype=np.uint8)
        varied = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        assert compute_histogram_entropy(uniform) < compute_histogram_entropy(varied)

    def test_percentile_range_uniform_zero(self):
        """Uniform image should have zero percentile range."""
        img = np.full((64, 64), 128, dtype=np.uint8)
        assert compute_percentile_range(img) == 0.0

    def test_percentile_range_varied_positive(self):
        """Varied image should have positive percentile range."""
        img = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        assert compute_percentile_range(img) > 0

    def test_center_vs_edge_ratio_uniform_one(self):
        """Uniform image should have ratio close to 1.0."""
        img = np.full((64, 64), 128, dtype=np.uint8)
        ratio = compute_center_vs_edge_ratio(img)
        assert abs(ratio - 1.0) < 0.01

    def test_center_vs_edge_ratio_vignette(self):
        """Vignette (dark edges) should have ratio > 1.0."""
        img = np.full((64, 64), 100, dtype=np.uint8)
        # Bright center
        img[16:48, 16:48] = 200
        ratio = compute_center_vs_edge_ratio(img)
        assert ratio > 1.0


class TestExtractFeatures:
    """Tests for the main extract_features function."""

    def test_returns_correct_shape(self):
        """Should return (n_images, n_features) array."""
        images = [np.random.randint(0, 255, (64, 64), dtype=np.uint8) for _ in range(5)]
        features = extract_features(images)

        assert features.shape[0] == 5
        assert features.shape[1] == len(get_feature_names())

    def test_consistent_feature_order(self):
        """Feature order should match get_feature_names()."""
        names = get_feature_names()
        assert len(names) == 8
        assert "mean_intensity" in names
        assert "gradient_magnitude" in names

    def test_reproducible(self):
        """Same image should produce same features."""
        img = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        features1 = extract_features([img])
        features2 = extract_features([img])

        np.testing.assert_array_equal(features1, features2)
