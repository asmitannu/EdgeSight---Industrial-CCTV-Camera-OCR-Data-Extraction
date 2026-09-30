"""
Settings and configuration parameters for the Stage 1 Industrial OCR pipeline.
Provides centralized management for OCR models, preprocessing options, confidence thresholds,
and file path configurations.
"""

import os
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class OCRConfig:
    """Configuration settings for PaddleOCR engine and recognition."""
    language: str = "en"
    use_gpu: bool = False
    use_angle_cls: bool = True  # Enable text orientation classifier (0/90/180/270 degrees)
    confidence_threshold: float = 0.50  # Minimum confidence to consider detection valid
    show_log: bool = False
    det_db_thresh: float = 0.3  # Detection score threshold
    det_db_box_thresh: float = 0.5  # Text box score threshold


@dataclass
class PreprocessingConfig:
    """Configuration settings for image enhancement and preprocessing pipeline."""
    # CLAHE parameters (Contrast Limited Adaptive Histogram Equalization)
    enable_clahe: bool = True
    clahe_clip_limit: float = 2.5
    clahe_tile_grid_size: Tuple[int, int] = (8, 8)

    # Upscaling parameters for small CCTV/distance text
    enable_upscaling: bool = True
    upscale_factor: float = 2.0
    min_dimension_for_upscale: int = 800  # Upscale if min(h, w) < this value

    # Geometric & Orientation corrections
    enable_deskew: bool = True
    max_deskew_angle: float = 45.0  # Max skew angle (degrees) to attempt correction
    enable_perspective_correction: bool = True
    perspective_min_contour_area_ratio: float = 0.05  # Min area ratio of image to consider a tag contour

    # Sharpening
    enable_sharpening: bool = True
    sharpen_strength: float = 1.5

    # Thresholding methods to include in variant generation
    enable_adaptive_thresh: bool = True
    enable_otsu_thresh: bool = True

    # Denoising
    enable_denoise: bool = True
    bilateral_d: int = 9
    bilateral_sigma_color: int = 75
    bilateral_sigma_space: int = 75


@dataclass
class VariantScoringConfig:
    """Weights for selecting the best preprocessing variant."""
    confidence_weight: float = 0.60
    alphanumeric_density_weight: float = 0.25
    word_count_weight: float = 0.15


@dataclass
class PipelineConfig:
    """Master pipeline configuration combining all modules."""
    ocr: OCRConfig = field(default_factory=OCRConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    scoring: VariantScoringConfig = field(default_factory=VariantScoringConfig)

    # Directory and File Settings
    input_dir: str = "input_images"
    output_dir: str = "output"
    debug_mode: bool = False

    # Supported image extensions
    supported_extensions: Tuple[str, ...] = (
        ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"
    )

    # Output subdirectory structure
    @property
    def csv_output_dir(self) -> str:
        return os.path.join(self.output_dir, "csv")

    @property
    def excel_output_dir(self) -> str:
        return os.path.join(self.output_dir, "excel")

    @property
    def processed_images_dir(self) -> str:
        return os.path.join(self.output_dir, "processed_images")

    @property
    def debug_output_dir(self) -> str:
        return os.path.join(self.output_dir, "debug")


# Default global instance
default_config = PipelineConfig()
