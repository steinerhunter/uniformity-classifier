"""
Feature extraction for uniformity test images.

Extracts hand-crafted features that capture what humans look for
when judging scanner uniformity:
- Intensity statistics (mean, std, coefficient of variation)
- Spatial analysis (gradient magnitude)
- Artifact detection (local variance hotspots)
- Histogram analysis (entropy, percentile range)
"""

from typing import List

import numpy as np
from scipy import ndimage


def compute_mean_intensity(image: np.ndarray) -> float:
    """Average pixel value - baseline brightness level."""
    return float(np.mean(image))


def compute_std_intensity(image: np.ndarray) -> float:
    """Standard deviation - overall variability."""
    return float(np.std(image))


def compute_coef_of_variation(image: np.ndarray) -> float:
    """Coefficient of variation (std/mean) - normalized uniformity measure."""
    mean = np.mean(image)
    if mean == 0:
        return 0.0
    return float(np.std(image) / mean)


def compute_gradient_magnitude(image: np.ndarray) -> float:
    """
    Average gradient magnitude using Sobel filter.
    Detects edges and transitions - uniform images have low gradients.
    """
    # Sobel filters for x and y directions
    sobel_x = ndimage.sobel(image.astype(np.float64), axis=1)
    sobel_y = ndimage.sobel(image.astype(np.float64), axis=0)

    # Magnitude of gradient
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

    return float(np.mean(magnitude))


def compute_max_local_variance(image: np.ndarray, window_size: int = 16) -> float:
    """
    Maximum local variance - detects artifact hotspots.
    Scans image with sliding window and finds highest variance region.
    """
    image = image.astype(np.float64)
    h, w = image.shape

    if h < window_size or w < window_size:
        # Image too small for windowing
        return float(np.var(image))

    max_var = 0.0
    step = window_size // 2  # 50% overlap

    for y in range(0, h - window_size + 1, step):
        for x in range(0, w - window_size + 1, step):
            window = image[y:y + window_size, x:x + window_size]
            local_var = np.var(window)
            if local_var > max_var:
                max_var = local_var

    return float(max_var)


def compute_histogram_entropy(image: np.ndarray, bins: int = 256) -> float:
    """
    Histogram entropy - measures distribution spread.
    Uniform images have lower entropy (concentrated histogram).
    """
    hist, _ = np.histogram(image.flatten(), bins=bins, range=(0, 255))
    hist = hist / hist.sum()  # Normalize to probabilities

    # Remove zeros to avoid log(0)
    hist = hist[hist > 0]

    entropy = -np.sum(hist * np.log2(hist))
    return float(entropy)


def compute_percentile_range(image: np.ndarray) -> float:
    """
    Range between 95th and 5th percentile.
    Robust spread measure, less sensitive to outliers than max-min.
    """
    p5 = np.percentile(image, 5)
    p95 = np.percentile(image, 95)
    return float(p95 - p5)


def compute_center_vs_edge_ratio(image: np.ndarray, margin_fraction: float = 0.25) -> float:
    """
    Ratio of center intensity to edge intensity.
    Detects vignetting (darker edges) or inverse vignetting.
    Values close to 1.0 indicate uniform brightness.
    """
    h, w = image.shape
    margin_h = int(h * margin_fraction)
    margin_w = int(w * margin_fraction)

    # Center region
    center = image[margin_h:h - margin_h, margin_w:w - margin_w]
    center_mean = np.mean(center)

    # Edge region (everything except center)
    mask = np.ones_like(image, dtype=bool)
    mask[margin_h:h - margin_h, margin_w:w - margin_w] = False
    edge_mean = np.mean(image[mask])

    if edge_mean == 0:
        return 1.0

    return float(center_mean / edge_mean)


def extract_single_image_features(image: np.ndarray) -> dict:
    """
    Extract all features from a single image.

    Args:
        image: 2D grayscale numpy array

    Returns:
        Dictionary of feature name -> value
    """
    return {
        "mean_intensity": compute_mean_intensity(image),
        "std_intensity": compute_std_intensity(image),
        "coef_of_variation": compute_coef_of_variation(image),
        "gradient_magnitude": compute_gradient_magnitude(image),
        "max_local_variance": compute_max_local_variance(image),
        "histogram_entropy": compute_histogram_entropy(image),
        "percentile_range": compute_percentile_range(image),
        "center_vs_edge_ratio": compute_center_vs_edge_ratio(image),
    }


def extract_features(images: List[np.ndarray]) -> np.ndarray:
    """
    Extract features from a list of images.

    Args:
        images: List of 2D grayscale numpy arrays

    Returns:
        2D numpy array of shape (n_images, n_features)
    """
    feature_dicts = [extract_single_image_features(img) for img in images]

    # Convert to array, maintaining consistent feature order
    feature_names = list(feature_dicts[0].keys())
    features = np.array([
        [fd[name] for name in feature_names]
        for fd in feature_dicts
    ])

    return features


def get_feature_names() -> List[str]:
    """Return list of feature names in extraction order."""
    return [
        "mean_intensity",
        "std_intensity",
        "coef_of_variation",
        "gradient_magnitude",
        "max_local_variance",
        "histogram_entropy",
        "percentile_range",
        "center_vs_edge_ratio",
    ]
