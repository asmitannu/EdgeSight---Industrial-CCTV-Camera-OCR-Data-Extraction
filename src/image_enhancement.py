"""
Image enhancement module for industrial CCTV and camera photograph OCR.
Implements specialized filters, contrast enhancement, denoising, sharpening,
morphological operations, and thresholding tailored for industrial equipment text.
"""

import logging
from typing import Tuple, Optional, Union
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ImageEnhancer:
    """Provides modular image enhancement techniques for OCR preprocessing."""

    @staticmethod
    def to_grayscale(image: np.ndarray) -> np.ndarray:
        """
        Converts image to single-channel 8-bit grayscale if it has 3 or 4 channels.
        """
        if image is None or image.size == 0:
            raise ValueError("Input image is empty or None")

        if len(image.shape) == 2:
            return image.copy()

        if len(image.shape) == 3:
            channels = image.shape[2]
            if channels == 3:
                return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            elif channels == 4:
                return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
            elif channels == 1:
                return image.squeeze(axis=2)

        raise ValueError(f"Unexpected image shape for grayscale conversion: {image.shape}")

    @classmethod
    def apply_clahe(
        cls,
        image: np.ndarray,
        clip_limit: float = 2.5,
        tile_grid_size: Tuple[int, int] = (8, 8)
    ) -> np.ndarray:
        """
        Applies Contrast Limited Adaptive Histogram Equalization (CLAHE).
        For color images, converts to LAB color space and applies CLAHE strictly to
        the Lightness (L) channel to enhance contrast without distorting color balance.
        For grayscale images, applies CLAHE directly.
        """
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

        if len(image.shape) == 2:
            return clahe.apply(image)

        if len(image.shape) == 3 and image.shape[2] == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            cl = clahe.apply(l)
            merged = cv2.merge((cl, a, b))
            return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

        # Fallback for other channel counts
        gray = cls.to_grayscale(image)
        return clahe.apply(gray)

    @classmethod
    def apply_histogram_equalization(cls, image: np.ndarray) -> np.ndarray:
        """
        Applies standard global histogram equalization.
        """
        gray = cls.to_grayscale(image)
        return cv2.equalizeHist(gray)

    @classmethod
    def adjust_brightness_contrast(
        cls,
        image: np.ndarray,
        alpha: float = 1.2,
        beta: int = 10
    ) -> np.ndarray:
        """
        Adjusts contrast (alpha: 1.0-3.0) and brightness (beta: 0-100).
        Uses cv2.convertScaleAbs for saturated arithmetic (clipping at 0-255).
        """
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    @classmethod
    def apply_gamma_correction(cls, image: np.ndarray, gamma: float = 1.2) -> np.ndarray:
        """
        Applies non-linear gamma correction to recover text in underexposed (gamma < 1.0)
        or overexposed (gamma > 1.0) industrial lighting conditions.
        """
        inv_gamma = 1.0 / max(gamma, 1e-5)
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)

    @classmethod
    def denoise(
        cls,
        image: np.ndarray,
        method: str = "bilateral",
        d: int = 9,
        sigma_color: int = 75,
        sigma_space: int = 75,
        ksize: int = 3
    ) -> np.ndarray:
        """
        Applies noise reduction filter.
        - 'bilateral': Edge-preserving smoothing (preferred for OCR text).
        - 'gaussian': Standard low-pass smoothing.
        - 'median': Removes salt-and-pepper noise common in CCTV sensors.
        """
        if method == "bilateral":
            # Bilateral filter works on color or grayscale
            return cv2.bilateralFilter(image, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)
        elif method == "gaussian":
            k = ksize if ksize % 2 != 0 else ksize + 1
            return cv2.GaussianBlur(image, (k, k), 0)
        elif method == "median":
            k = ksize if ksize % 2 != 0 else ksize + 1
            return cv2.medianBlur(image, k)
        else:
            logger.warning(f"Unknown denoising method '{method}', returning original image")
            return image

    @classmethod
    def sharpen(
        cls,
        image: np.ndarray,
        strength: float = 1.5,
        use_unsharp_mask: bool = True
    ) -> np.ndarray:
        """
        Sharpens text edges.
        Uses Unsharp Masking (subtracting blurred version from original) or laplacian kernel.
        """
        if use_unsharp_mask:
            # Unsharp mask: image + strength * (image - gaussian_blur)
            blurred = cv2.GaussianBlur(image, (0, 0), 3)
            sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
            return np.clip(sharpened, 0, 255).astype(np.uint8)
        else:
            kernel = np.array([
                [0, -1, 0],
                [-1, 4 + strength, -1],
                [0, -1, 0]
            ], dtype=np.float32)
            return cv2.filter2D(image, -1, kernel)

    @classmethod
    def upscale(
        cls,
        image: np.ndarray,
        scale_factor: float = 2.0,
        interpolation: int = cv2.INTER_CUBIC
    ) -> np.ndarray:
        """
        Upscales small text regions using bicubic or lanczos interpolation.
        Essential for distant CCTV camera captures with low character resolution.
        """
        if scale_factor <= 1.0:
            return image

        h, w = image.shape[:2]
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)

        return cv2.resize(image, (new_w, new_h), interpolation=interpolation)

    @classmethod
    def apply_morphology(
        cls,
        binary_image: np.ndarray,
        operation: str = "close",
        kernel_size: Tuple[int, int] = (3, 3)
    ) -> np.ndarray:
        """
        Applies morphological operations on binary/grayscale images:
        - 'close': Dilation followed by Erosion (bridges breaks in dot-matrix / stamped text).
        - 'open': Erosion followed by Dilation (removes speckles and isolated noise).
        - 'dilate': Expands text strokes (helpful for thin/faded fonts).
        - 'erode': Thins text strokes (helpful for bleeding/blurry ink).
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)

        op_map = {
            "close": cv2.MORPH_CLOSE,
            "open": cv2.MORPH_OPEN,
            "dilate": cv2.MORPH_DILATE,
            "erode": cv2.MORPH_ERODE,
            "gradient": cv2.MORPH_GRADIENT,
            "tophat": cv2.MORPH_TOPHAT,
            "blackhat": cv2.MORPH_BLACKHAT,
        }

        op_code = op_map.get(operation.lower())
        if op_code is None:
            raise ValueError(f"Unknown morphology operation: {operation}. Choose from {list(op_map.keys())}")

        if op_code == cv2.MORPH_DILATE:
            return cv2.dilate(binary_image, kernel, iterations=1)
        elif op_code == cv2.MORPH_ERODE:
            return cv2.erode(binary_image, kernel, iterations=1)
        else:
            return cv2.morphologyEx(binary_image, op_code, kernel)

    @classmethod
    def apply_threshold(
        cls,
        image: np.ndarray,
        method: str = "adaptive",
        block_size: int = 15,
        c_val: int = 4
    ) -> np.ndarray:
        """
        Applies thresholding to produce high-contrast binary text image:
        - 'adaptive': Adaptive Gaussian thresholding (handles uneven lighting across nameplates).
        - 'otsu': Otsu bimodal automatic global thresholding.
        - 'global': Standard fixed-value threshold at 127.
        """
        gray = cls.to_grayscale(image)

        if method == "adaptive":
            # block_size must be odd and > 1
            bs = block_size if block_size % 2 != 0 else block_size + 1
            return cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                bs,
                c_val
            )
        elif method == "otsu":
            # Apply light Gaussian blur before Otsu to remove high-frequency noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return thresh
        elif method == "global":
            _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            return thresh
        else:
            raise ValueError(f"Unknown thresholding method '{method}'. Choose from 'adaptive', 'otsu', 'global'")
