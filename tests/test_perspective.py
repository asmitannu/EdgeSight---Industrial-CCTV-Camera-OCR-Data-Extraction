"""
Tests for PerspectiveCorrector module.
"""

import pytest
import numpy as np
import cv2

from src.perspective_correction import PerspectiveCorrector


def test_order_points():
    # Randomly ordered 4 points of a rectangle (10, 10) to (100, 50)
    pts = np.array([[100, 10], [10, 50], [10, 10], [100, 50]], dtype="float32")
    ordered = PerspectiveCorrector.order_points(pts)

    # Top-left should be (10, 10)
    assert np.allclose(ordered[0], [10, 10])
    # Top-right should be (100, 10)
    assert np.allclose(ordered[1], [100, 10])
    # Bottom-right should be (100, 50)
    assert np.allclose(ordered[2], [100, 50])
    # Bottom-left should be (10, 50)
    assert np.allclose(ordered[3], [10, 50])


def test_rotate_image():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    rotated = PerspectiveCorrector.rotate_image(img, angle=45.0)

    assert rotated.shape[0] > 100
    assert rotated.shape[1] > 100


def test_four_point_transform():
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    # Draw a tilted square
    pts = np.array([[50, 50], [250, 60], [240, 260], [60, 250]], dtype="float32")

    warped = PerspectiveCorrector.four_point_transform(img, pts)
    assert warped.shape[0] > 0
    assert warped.shape[1] > 0


def test_deskew_no_text():
    blank = np.full((100, 100, 3), 255, dtype=np.uint8)
    deskewed, angle = PerspectiveCorrector.deskew(blank)
    assert angle == 0.0
    assert deskewed.shape == blank.shape
