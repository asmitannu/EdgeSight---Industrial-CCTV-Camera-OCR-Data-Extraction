"""
Image loader module for the Industrial OCR system.
Provides robust image loading (Unicode/Windows safe), validation, directory scanning,
and Region of Interest (ROI) extraction.
"""

import os
import logging
from typing import List, Tuple, Optional, Dict, Any
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ImageLoader:
    """Handles loading, validation, and ROI cropping for image files."""

    SUPPORTED_EXTENSIONS = (
        ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"
    )

    @classmethod
    def load_image(cls, image_path: str) -> np.ndarray:
        """
        Loads an image from disk safely, handling Windows Unicode paths and special characters.

        Args:
            image_path: Absolute or relative path to the image file.

        Returns:
            np.ndarray: BGR image array.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file cannot be decoded as an image or is corrupted.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        if not os.path.isfile(image_path):
            raise ValueError(f"Path is not a valid file: {image_path}")

        # Check extension
        ext = os.path.splitext(image_path)[1].lower()
        if ext not in cls.SUPPORTED_EXTENSIONS:
            logger.warning(f"File extension '{ext}' not in standard list {cls.SUPPORTED_EXTENSIONS}")

        # Read binary using numpy to support unicode/special character paths on Windows
        try:
            with open(image_path, "rb") as f:
                file_bytes = np.frombuffer(f.read(), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        except Exception as e:
            raise ValueError(f"Failed to read image bytes from '{image_path}': {e}") from e

        if image is None:
            raise ValueError(f"Image decoder failed (corrupted or unsupported format): {image_path}")

        cls.validate_image(image, image_path)
        return image

    @classmethod
    def validate_image(cls, image: np.ndarray, source_name: str = "image") -> None:
        """
        Validates an image matrix.

        Args:
            image: Image ndarray.
            source_name: Descriptive name/path for logging.

        Raises:
            ValueError: If image is empty or invalid.
        """
        if image is None or not isinstance(image, np.ndarray):
            raise ValueError(f"Invalid image array for {source_name}: expected np.ndarray, got {type(image)}")

        if image.size == 0:
            raise ValueError(f"Image array is empty (0 bytes) for {source_name}")

        if len(image.shape) < 2:
            raise ValueError(f"Image must have at least 2 dimensions, got shape {image.shape} for {source_name}")

        h, w = image.shape[:2]
        if h <= 0 or w <= 0:
            raise ValueError(f"Invalid image dimensions ({w}x{h}) for {source_name}")

        logger.debug(f"Validated {source_name}: shape={image.shape}, dtype={image.dtype}")

    @classmethod
    def crop_roi(
        cls,
        image: np.ndarray,
        roi: Optional[Tuple[int, int, int, int]] = None
    ) -> np.ndarray:
        """
        Crops a Region of Interest (ROI) from an image with boundary validation.

        Args:
            image: Input image ndarray (H, W, C) or (H, W).
            roi: Optional tuple of (x, y, width, height). If None, returns original image.

        Returns:
            np.ndarray: Cropped image region.

        Raises:
            ValueError: If ROI is outside image bounds or resulting crop is empty.
        """
        if roi is None:
            return image

        cls.validate_image(image, "ROI input")

        img_h, img_w = image.shape[:2]
        x, y, w, h = roi

        if w <= 0 or h <= 0:
            raise ValueError(f"ROI width and height must be positive, got width={w}, height={h}")

        # Clamp boundaries safely
        x1 = max(0, min(x, img_w - 1))
        y1 = max(0, min(y, img_h - 1))
        x2 = max(x1 + 1, min(x + w, img_w))
        y2 = max(y1 + 1, min(y + h, img_h))

        cropped = image[y1:y2, x1:x2].copy()
        if cropped.size == 0:
            raise ValueError(f"Cropped ROI resulted in 0-size image from bounds ({x1},{y1},{x2},{y2})")

        logger.info(f"Cropped ROI: [x={x1}, y={y1}, w={x2-x1}, h={y2-y1}] from original ({img_w}x{img_h})")
        return cropped

    @classmethod
    def parse_roi_string(cls, roi_str: Optional[str]) -> Optional[Tuple[int, int, int, int]]:
        """
        Parses ROI string in 'x,y,w,h' format.

        Args:
            roi_str: String such as "100,50,400,300".

        Returns:
            Optional tuple (x, y, w, h).
        """
        if not roi_str or not roi_str.strip():
            return None

        parts = [p.strip() for p in roi_str.split(",")]
        if len(parts) != 4:
            raise ValueError(f"ROI string must have 4 comma-separated integers (x,y,w,h), got: '{roi_str}'")

        try:
            x, y, w, h = [int(p) for p in parts]
            return (x, y, w, h)
        except ValueError as e:
            raise ValueError(f"Failed to parse ROI coordinates as integers from '{roi_str}': {e}") from e

    @classmethod
    def get_image_files(
        cls,
        target_path: str,
        supported_extensions: Tuple[str, ...] = SUPPORTED_EXTENSIONS
    ) -> List[str]:
        """
        Finds all image files from a given file path or directory.

        Args:
            target_path: Path to a single image or directory.
            supported_extensions: Tuple of supported file extensions.

        Returns:
            List of valid image file paths.

        Raises:
            FileNotFoundError: If the target path does not exist.
        """
        if not os.path.exists(target_path):
            raise FileNotFoundError(f"Input path not found: {target_path}")

        valid_extensions = tuple(ext.lower() for ext in supported_extensions)

        if os.path.isfile(target_path):
            ext = os.path.splitext(target_path)[1].lower()
            if ext in valid_extensions:
                return [os.path.abspath(target_path)]
            else:
                logger.warning(f"File {target_path} extension {ext} not in supported list {valid_extensions}")
                return [os.path.abspath(target_path)]

        # If directory, scan for supported images
        image_files = []
        for root, _, files in os.walk(target_path):
            for file in files:
                if file.lower().endswith(valid_extensions):
                    image_files.append(os.path.abspath(os.path.join(root, file)))

        image_files.sort()
        return image_files
