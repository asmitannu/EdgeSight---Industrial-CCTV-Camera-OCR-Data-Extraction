"""
Tests for ImageEnhancer and PreprocessingPipeline modules.
"""

import pytest
import numpy as np
import cv2

from config.settings import PreprocessingConfig
from src.image_enhancement import ImageEnhancer
from src.preprocessing import PreprocessingPipeline


@pytest.fixture
def sample_bgr_image():
    img = np.full((100, 100, 3), 128, dtype=np.uint8)
    cv2.putText(img, "TEST", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return img


@pytest.fixture
def sample_gray_image():
    img = np.full((100, 100), 128, dtype=np.uint8)
    cv2.putText(img, "TEST", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 255, 2)
    return img


def test_to_grayscale(sample_bgr_image, sample_gray_image):
    gray_from_bgr = ImageEnhancer.to_grayscale(sample_bgr_image)
    assert len(gray_from_bgr.shape) == 2
    assert gray_from_bgr.shape == (100, 100)

    gray_from_gray = ImageEnhancer.to_grayscale(sample_gray_image)
    assert gray_from_gray.shape == (100, 100)


def test_apply_clahe(sample_bgr_image, sample_gray_image):
    clahe_bgr = ImageEnhancer.apply_clahe(sample_bgr_image)
    assert clahe_bgr.shape == sample_bgr_image.shape

    clahe_gray = ImageEnhancer.apply_clahe(sample_gray_image)
    assert clahe_gray.shape == sample_gray_image.shape


def test_denoise(sample_bgr_image):
    bilateral = ImageEnhancer.denoise(sample_bgr_image, method="bilateral")
    assert bilateral.shape == sample_bgr_image.shape

    gaussian = ImageEnhancer.denoise(sample_bgr_image, method="gaussian")
    assert gaussian.shape == sample_bgr_image.shape

    median = ImageEnhancer.denoise(sample_bgr_image, method="median")
    assert median.shape == sample_bgr_image.shape


def test_sharpen(sample_bgr_image):
    sharpened = ImageEnhancer.sharpen(sample_bgr_image, strength=1.5)
    assert sharpened.shape == sample_bgr_image.shape
    assert sharpened.dtype == np.uint8


def test_upscale(sample_bgr_image):
    upscaled = ImageEnhancer.upscale(sample_bgr_image, scale_factor=2.0)
    assert upscaled.shape == (200, 200, 3)


def test_apply_morphology(sample_gray_image):
    closed = ImageEnhancer.apply_morphology(sample_gray_image, operation="close")
    assert closed.shape == sample_gray_image.shape

    opened = ImageEnhancer.apply_morphology(sample_gray_image, operation="open")
    assert opened.shape == sample_gray_image.shape


def test_apply_threshold(sample_bgr_image):
    adaptive = ImageEnhancer.apply_threshold(sample_bgr_image, method="adaptive")
    assert len(adaptive.shape) == 2
    assert np.array_equal(np.unique(adaptive), np.array([0, 255])) or len(np.unique(adaptive)) <= 2

    otsu = ImageEnhancer.apply_threshold(sample_bgr_image, method="otsu")
    assert len(otsu.shape) == 2


def test_preprocessing_pipeline_variants(sample_bgr_image):
    config = PreprocessingConfig()
    pipeline = PreprocessingPipeline(config)

    variants = pipeline.generate_variants(sample_bgr_image, base_name="test_sample")
    assert "original" in variants
    assert "enhanced_clahe" in variants
    assert "adaptive_thresh" in variants
    assert "otsu_thresh" in variants
    assert len(variants) >= 4
