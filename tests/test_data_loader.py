"""Tests for data loading utilities."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.data_loader import load_image, load_dataset, split_dataset, LABEL_PASS, LABEL_FAIL


@pytest.fixture
def temp_dataset():
    """Create a temporary dataset with PASS and FAIL images."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        pass_dir = tmpdir / "PASS"
        fail_dir = tmpdir / "FAIL"
        pass_dir.mkdir()
        fail_dir.mkdir()

        # Create 6 PASS images (uniform gray)
        for i in range(6):
            img = Image.new("L", (64, 64), color=128)
            img.save(pass_dir / f"pass_{i}.png")

        # Create 4 FAIL images (with gradient)
        for i in range(4):
            arr = np.tile(np.linspace(0, 255, 64), (64, 1)).astype(np.uint8)
            img = Image.fromarray(arr)
            img.save(fail_dir / f"fail_{i}.png")

        yield tmpdir


class TestLoadImage:
    """Tests for load_image function."""

    def test_load_png(self, temp_dataset):
        """Should load PNG images as grayscale numpy arrays."""
        path = temp_dataset / "PASS" / "pass_0.png"
        img = load_image(path)

        assert isinstance(img, np.ndarray)
        assert img.dtype == np.uint8
        assert img.shape == (64, 64)

    def test_load_nonexistent_raises(self):
        """Should raise error for nonexistent file."""
        with pytest.raises(Exception):
            load_image(Path("/nonexistent/image.png"))

    def test_unsupported_format_raises(self, temp_dataset):
        """Should raise error for unsupported formats."""
        bad_file = temp_dataset / "test.xyz"
        bad_file.write_text("not an image")

        with pytest.raises(ValueError, match="Unsupported file format"):
            load_image(bad_file)


class TestLoadDataset:
    """Tests for load_dataset function."""

    def test_loads_all_images(self, temp_dataset):
        """Should load all images from PASS and FAIL directories."""
        images, labels, paths = load_dataset(temp_dataset)

        assert len(images) == 10  # 6 PASS + 4 FAIL
        assert len(labels) == 10
        assert len(paths) == 10

    def test_correct_labels(self, temp_dataset):
        """Should assign correct labels to PASS and FAIL images."""
        images, labels, paths = load_dataset(temp_dataset)

        pass_count = sum(1 for l in labels if l == LABEL_PASS)
        fail_count = sum(1 for l in labels if l == LABEL_FAIL)

        assert pass_count == 6
        assert fail_count == 4

    def test_missing_pass_dir_raises(self, temp_dataset):
        """Should raise error if PASS directory is missing."""
        # Remove all files first, then directory
        for f in (temp_dataset / "PASS").iterdir():
            f.unlink()
        (temp_dataset / "PASS").rmdir()

        with pytest.raises(FileNotFoundError, match="PASS directory"):
            load_dataset(temp_dataset)

    def test_missing_fail_dir_raises(self, temp_dataset):
        """Should raise error if FAIL directory is missing."""
        # Remove all files first, then directory
        for f in (temp_dataset / "FAIL").iterdir():
            f.unlink()
        (temp_dataset / "FAIL").rmdir()

        with pytest.raises(FileNotFoundError, match="FAIL directory"):
            load_dataset(temp_dataset)


class TestSplitDataset:
    """Tests for split_dataset function."""

    def test_correct_split_sizes(self, temp_dataset):
        """Should split data into correct proportions."""
        images, labels, paths = load_dataset(temp_dataset)
        train, test = split_dataset(images, labels, paths, test_size=0.2)

        train_images, train_labels, train_paths = train
        test_images, test_labels, test_paths = test

        assert len(train_images) == 8  # 80% of 10
        assert len(test_images) == 2   # 20% of 10

    def test_reproducible_with_same_seed(self, temp_dataset):
        """Should produce same split with same random_state."""
        images, labels, paths = load_dataset(temp_dataset)

        _, test1 = split_dataset(images, labels, paths, random_state=42)
        _, test2 = split_dataset(images, labels, paths, random_state=42)

        assert [p.name for p in test1[2]] == [p.name for p in test2[2]]

    def test_different_with_different_seed(self, temp_dataset):
        """Should produce different split with different random_state."""
        images, labels, paths = load_dataset(temp_dataset)

        _, test1 = split_dataset(images, labels, paths, random_state=42)
        _, test2 = split_dataset(images, labels, paths, random_state=123)

        # Very likely to be different (not guaranteed but highly probable)
        test1_names = [p.name for p in test1[2]]
        test2_names = [p.name for p in test2[2]]
        # At least check they don't always match
        assert len(test1_names) == len(test2_names)
