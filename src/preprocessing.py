"""
Preprocessing pipeline orchestrator.
Generates multiple intelligent image preprocessing variants (contrast enhancement,
thresholding, deskewing, perspective correction, upscaling) for the multi-variant OCR strategy.
"""

import os
import logging
from typing import Dict, Optional, Tuple
import cv2
import numpy as np

from config.settings import PreprocessingConfig
from src.image_enhancement import ImageEnhancer
from src.perspective_correction import PerspectiveCorrector

logger = logging.getLogger(__name__)


class PreprocessingPipeline:
    """Orchestrates image preprocessing and generates variants for OCR evaluation."""

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or PreprocessingConfig()

    def generate_variants(
        self,
        image: np.ndarray,
        base_name: str = "image"
    ) -> Dict[str, np.ndarray]:
        """
        Generates a dictionary of named preprocessing variants for an image.

        Variants:
            - 'original': Base image
            - 'enhanced_clahe': CLAHE in LAB space + mild sharpening
            - 'adaptive_thresh': Bilateral filter + Adaptive Gaussian binarization + morphological close
            - 'otsu_thresh': Grayscale + Otsu binarization
            - 'upscaled_sharpened': 2x bicubic upscale + unsharp mask sharpening
            - 'deskewed': Rotation corrected to level text lines (if skew > 0.5°)
            - 'perspective_corrected': 4-point perspective transformed (if tag contour found)

        Args:
            image: Input image array (BGR or Grayscale).
            base_name: Name identifier for logging.

        Returns:
            Dict[str, np.ndarray]: Mapping from variant_name -> processed image.
        """
        variants: Dict[str, np.ndarray] = {}

        # 1. Original
        variants["original"] = image.copy()

        # 2. CLAHE Enhanced (essential for uneven CCTV lighting and shadowed labels)
        if self.config.enable_clahe:
            try:
                clahe_img = ImageEnhancer.apply_clahe(
                    image,
                    clip_limit=self.config.clahe_clip_limit,
                    tile_grid_size=self.config.clahe_tile_grid_size
                )
                if self.config.enable_sharpening:
                    clahe_img = ImageEnhancer.sharpen(clahe_img, strength=1.0)
                variants["enhanced_clahe"] = clahe_img
            except Exception as e:
                logger.warning(f"Failed to generate CLAHE variant for {base_name}: {e}")

        # 3. Adaptive Thresholding (great for stamped/laser-etched industrial text)
        if self.config.enable_adaptive_thresh:
            try:
                # Bilateral filter first to preserve edges while smoothing background noise
                denoised = ImageEnhancer.denoise(
                    image,
                    method="bilateral",
                    d=self.config.bilateral_d,
                    sigma_color=self.config.bilateral_sigma_color,
                    sigma_space=self.config.bilateral_sigma_space
                )
                adaptive_bin = ImageEnhancer.apply_threshold(denoised, method="adaptive", block_size=15, c_val=4)
                # Apply morphological closing to bridge broken dot-matrix / stamped character strokes
                adaptive_bin = ImageEnhancer.apply_morphology(adaptive_bin, operation="close", kernel_size=(3, 3))
                # Convert back to 3-channel for PaddleOCR compatibility
                variants["adaptive_thresh"] = cv2.cvtColor(adaptive_bin, cv2.COLOR_GRAY2BGR)
            except Exception as e:
                logger.warning(f"Failed to generate adaptive threshold variant for {base_name}: {e}")

        # 4. Otsu Thresholding (clean bimodal separation for high-contrast plates)
        if self.config.enable_otsu_thresh:
            try:
                otsu_bin = ImageEnhancer.apply_threshold(image, method="otsu")
                variants["otsu_thresh"] = cv2.cvtColor(otsu_bin, cv2.COLOR_GRAY2BGR)
            except Exception as e:
                logger.warning(f"Failed to generate Otsu threshold variant for {base_name}: {e}")

        # 5. Upscaled & Sharpened (improves distant CCTV serial number recognition)
        if self.config.enable_upscaling:
            try:
                h, w = image.shape[:2]
                if min(h, w) < self.config.min_dimension_for_upscale or self.config.upscale_factor > 1.0:
                    upscaled = ImageEnhancer.upscale(image, scale_factor=self.config.upscale_factor)
                    if self.config.enable_sharpening:
                        upscaled = ImageEnhancer.sharpen(upscaled, strength=self.config.sharpen_strength)
                    variants["upscaled_sharpened"] = upscaled
            except Exception as e:
                logger.warning(f"Failed to generate upscaled variant for {base_name}: {e}")

        # 6. Deskewed (straightens tilted text)
        if self.config.enable_deskew:
            try:
                deskewed_img, angle = PerspectiveCorrector.deskew(
                    image,
                    max_angle=self.config.max_deskew_angle
                )
                if abs(angle) >= 0.5:
                    variants["deskewed"] = deskewed_img
                    logger.debug(f"Added deskewed variant ({angle:.1f}°) for {base_name}")
            except Exception as e:
                logger.warning(f"Failed to generate deskewed variant for {base_name}: {e}")

        # 7. Perspective Corrected (corrects angled CCTV camera view if tag contour detected)
        if self.config.enable_perspective_correction:
            try:
                perspective_img, corrected = PerspectiveCorrector.correct_perspective_if_detected(
                    image,
                    min_area_ratio=self.config.perspective_min_contour_area_ratio
                )
                if corrected:
                    variants["perspective_corrected"] = perspective_img
                    logger.info(f"Added 4-point perspective corrected variant for {base_name}")
            except Exception as e:
                logger.warning(f"Failed to generate perspective corrected variant for {base_name}: {e}")

        logger.info(f"Generated {len(variants)} preprocessing variants for '{base_name}': {list(variants.keys())}")
        return variants

    @staticmethod
    def save_variants(
        variants: Dict[str, np.ndarray],
        output_dir: str,
        base_name: str
    ) -> None:
        """
        Saves all generated preprocessing variants to disk for inspection/debugging.

        Args:
            variants: Mapping of variant_name -> image ndarray.
            output_dir: Directory where images should be stored.
            base_name: File stem (e.g. 'sample_01').
        """
        os.makedirs(output_dir, exist_ok=True)
        stem = os.path.splitext(base_name)[0]

        for variant_name, var_img in variants.items():
            out_filename = f"{stem}_{variant_name}.png"
            out_path = os.path.join(output_dir, out_filename)
            try:
                # Use cv2.imencode for safe Unicode path writing on Windows
                ext = ".png"
                success, encoded = cv2.imencode(ext, var_img)
                if success:
                    with open(out_path, "wb") as f:
                        f.write(encoded)
                    logger.debug(f"Saved debug variant: {out_path}")
            except Exception as e:
                logger.warning(f"Failed to save debug variant '{out_path}': {e}")
