"""
Data loading utilities for uniformity test images.

Supports:
- DICOM files (.dcm)
- Standard image formats (.png, .jpg, .jpeg)

Expected directory structure:
    data/
    ├── PASS/
    │   └── *.dcm or *.png
    └── FAIL/
        └── *.dcm or *.png
"""

from pathlib import Path
from typing import Tuple, List

import numpy as np
from PIL import Image

# Optional DICOM support
try:
    import pydicom
    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False


# Labels: PASS=0, FAIL=1
LABEL_PASS = 0
LABEL_FAIL = 1


def load_image(path: Path) -> np.ndarray:
    """
    Load a single image from disk.

    Args:
        path: Path to image file (.dcm, .png, .jpg)

    Returns:
        2D numpy array (grayscale) with values normalized to 0-255
    """
    suffix = path.suffix.lower()

    if suffix == ".dcm":
        if not HAS_PYDICOM:
            raise ImportError("pydicom required for DICOM files: pip install pydicom")
        dcm = pydicom.dcmread(path)
        pixel_array = dcm.pixel_array.astype(np.float32)
        # Normalize to 0-255
        if pixel_array.max() > pixel_array.min():
            pixel_array = (pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min()) * 255
        return pixel_array.astype(np.uint8)

    elif suffix in {".png", ".jpg", ".jpeg"}:
        img = Image.open(path).convert("L")  # Convert to grayscale
        return np.array(img)

    else:
        raise ValueError(f"Unsupported file format: {suffix}")


def load_dataset(data_dir: Path) -> Tuple[List[np.ndarray], List[int], List[Path]]:
    """
    Load all images from PASS and FAIL directories.

    Args:
        data_dir: Root data directory containing PASS/ and FAIL/ subdirs

    Returns:
        Tuple of (images, labels, paths)
        - images: List of 2D numpy arrays
        - labels: List of ints (0=PASS, 1=FAIL)
        - paths: List of Path objects for each image
    """
    data_dir = Path(data_dir)
    pass_dir = data_dir / "PASS"
    fail_dir = data_dir / "FAIL"

    if not pass_dir.exists():
        raise FileNotFoundError(f"PASS directory not found: {pass_dir}")
    if not fail_dir.exists():
        raise FileNotFoundError(f"FAIL directory not found: {fail_dir}")

    images = []
    labels = []
    paths = []

    valid_extensions = {".dcm", ".png", ".jpg", ".jpeg"}

    # Load PASS images
    for path in sorted(pass_dir.iterdir()):
        if path.suffix.lower() in valid_extensions:
            images.append(load_image(path))
            labels.append(LABEL_PASS)
            paths.append(path)

    # Load FAIL images
    for path in sorted(fail_dir.iterdir()):
        if path.suffix.lower() in valid_extensions:
            images.append(load_image(path))
            labels.append(LABEL_FAIL)
            paths.append(path)

    if len(images) == 0:
        raise ValueError(f"No valid images found in {data_dir}")

    return images, labels, paths


def split_dataset(
    images: List[np.ndarray],
    labels: List[int],
    paths: List[Path],
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[Tuple[List[np.ndarray], List[int], List[Path]],
           Tuple[List[np.ndarray], List[int], List[Path]]]:
    """
    Split dataset into train and test sets with stratification.

    Args:
        images: List of image arrays
        labels: List of labels
        paths: List of file paths
        test_size: Fraction of data to use for testing (default 0.2)
        random_state: Random seed for reproducibility

    Returns:
        Tuple of ((train_images, train_labels, train_paths),
                  (test_images, test_labels, test_paths))
    """
    from sklearn.model_selection import train_test_split

    # Create indices for splitting
    indices = list(range(len(images)))

    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=labels  # Maintain class balance
    )

    train_images = [images[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    train_paths = [paths[i] for i in train_idx]

    test_images = [images[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]
    test_paths = [paths[i] for i in test_idx]

    return (train_images, train_labels, train_paths), (test_images, test_labels, test_paths)
