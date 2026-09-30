"""
Tests for ImageLoader module.
"""

import os
import tempfile
import pytest
import numpy as np
import cv2

from src.image_loader import ImageLoader


@pytest.fixture
def sample_image():
    """Creates a simple 200x200 3-channel test image."""
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.putText(img, "TEST", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return img


@pytest.fixture
def temp_image_file(sample_image):
    """Saves sample image to a temporary file."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        temp_path = f.name
    cv2.imwrite(temp_path, sample_image)
    yield temp_path
    if os.path.exists(temp_path):
        os.remove(temp_path)


def test_load_valid_image(temp_image_file):
    img = ImageLoader.load_image(temp_image_file)
    assert isinstance(img, np.ndarray)
    assert img.shape == (200, 200, 3)


def test_load_nonexistent_image():
    with pytest.raises(FileNotFoundError):
        ImageLoader.load_image("non_existent_image_12345.png")


def test_validate_image_valid(sample_image):
    # Should not raise
    ImageLoader.validate_image(sample_image)


def test_validate_image_invalid():
    with pytest.raises(ValueError):
        ImageLoader.validate_image(np.array([]))

    with pytest.raises(ValueError):
        ImageLoader.validate_image(None)


def test_crop_roi_valid(sample_image):
    roi = (20, 30, 100, 80)
    cropped = ImageLoader.crop_roi(sample_image, roi)
    assert cropped.shape == (80, 100, 3)


def test_crop_roi_none(sample_image):
    cropped = ImageLoader.crop_roi(sample_image, None)
    assert cropped.shape == sample_image.shape


def test_crop_roi_out_of_bounds(sample_image):
    # ROI extending beyond image boundaries
    roi = (150, 150, 100, 100)
    cropped = ImageLoader.crop_roi(sample_image, roi)
    assert cropped.shape[0] == 50
    assert cropped.shape[1] == 50


def test_parse_roi_string():
    assert ImageLoader.parse_roi_string("10,20,30,40") == (10, 20, 30, 40)
    assert ImageLoader.parse_roi_string("  100 , 50 , 200 , 150 ") == (100, 50, 200, 150)
    assert ImageLoader.parse_roi_string(None) is None
    assert ImageLoader.parse_roi_string("") is None

    with pytest.raises(ValueError):
        ImageLoader.parse_roi_string("10,20,30")

    with pytest.raises(ValueError):
        ImageLoader.parse_roi_string("a,b,c,d")


def test_get_image_files(temp_image_file):
    files = ImageLoader.get_image_files(temp_image_file)
    assert len(files) == 1
    assert files[0] == os.path.abspath(temp_image_file)

    temp_dir = os.path.dirname(temp_image_file)
    files_in_dir = ImageLoader.get_image_files(temp_dir)
    assert len(files_in_dir) >= 1
