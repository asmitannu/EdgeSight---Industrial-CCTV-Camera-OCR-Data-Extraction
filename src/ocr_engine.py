"""
OCR Engine module wrapping PaddleOCR.
Handles model loading, inference, bounding box extraction, confidence thresholding,
and standardized structured output.
"""

import logging
from typing import Dict, List, Any, Optional, Union
import numpy as np

from config.settings import OCRConfig

logger = logging.getLogger(__name__)


class PaddleOCREngine:
    """Wrapper around PaddleOCR for text detection and recognition."""

    def __init__(self, config: Optional[OCRConfig] = None):
        self.config = config or OCRConfig()
        self._ocr_instance = None
        self._initialized = False

    def _initialize_engine(self) -> None:
        """Lazily initializes the PaddleOCR model."""
        if self._initialized:
            return

        try:
            from paddleocr import PaddleOCR

            logger.info(
                f"Initializing PaddleOCR (lang='{self.config.language}', "
                f"use_angle_cls={self.config.use_angle_cls}, "
                f"use_gpu={self.config.use_gpu})..."
            )
            self._ocr_instance = PaddleOCR(
                use_angle_cls=self.config.use_angle_cls,
                lang=self.config.language,
                use_gpu=self.config.use_gpu,
                show_log=self.config.show_log,
                det_db_thresh=self.config.det_db_thresh,
                det_db_box_thresh=self.config.det_db_box_thresh,
            )
            self._initialized = True
            logger.info("PaddleOCR engine initialized successfully.")
        except ImportError as e:
            logger.error(
                "PaddleOCR is not installed. Please install with: pip install paddlepaddle paddleocr"
            )
            raise ImportError(f"Missing required PaddleOCR dependencies: {e}") from e
        except Exception as e:
            logger.error(f"Failed to initialize PaddleOCR: {e}")
            raise RuntimeError(f"PaddleOCR initialization failed: {e}") from e

    def extract_text(
        self,
        image: np.ndarray,
        image_name: str = "unknown.jpg",
        variant_name: str = "original"
    ) -> Dict[str, Any]:
        """
        Executes OCR inference on an image matrix.

        Args:
            image: BGR numpy ndarray.
            image_name: Name of the input image file.
            variant_name: Name of the preprocessing variant being evaluated.

        Returns:
            Dict containing:
                - 'image_name': str
                - 'variant_name': str
                - 'detections': List of dicts with:
                    - 'text': Extracted text string
                    - 'confidence': Float confidence score (0.0 to 1.0)
                    - 'bounding_box': 4 polygon vertices [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
                    - 'box_flat': [x1, y1, x2, y2, x3, y3, x4, y4]
                    - 'is_confident': Bool indicating whether confidence >= threshold
        """
        self._initialize_engine()

        result_payload: Dict[str, Any] = {
            "image_name": image_name,
            "variant_name": variant_name,
            "detections": []
        }

        if image is None or image.size == 0:
            logger.warning(f"Empty image passed to OCR for {image_name} [{variant_name}]")
            return result_payload

        # Ensure 3-channel image for PaddleOCR
        if len(image.shape) == 2:
            import cv2
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        try:
            # PaddleOCR returns: [ [ [box_coords], (text, confidence) ], ... ]
            raw_results = self._ocr_instance.ocr(image, cls=self.config.use_angle_cls)
        except Exception as e:
            logger.error(f"OCR inference failed for {image_name} [{variant_name}]: {e}")
            return result_payload

        if not raw_results or raw_results[0] is None:
            logger.debug(f"No text detected for {image_name} [{variant_name}]")
            return result_payload

        detections = []
        for line in raw_results[0]:
            try:
                box_coords = line[0]  # List of 4 points [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                text_info = line[1]   # Tuple: (text_str, confidence_float)

                raw_text = str(text_info[0]).strip()
                confidence = float(text_info[1])

                # Flatten box coords for easy serialization
                flat_coords = []
                for pt in box_coords:
                    flat_coords.extend([round(float(pt[0]), 1), round(float(pt[1]), 1)])

                rounded_box = [[round(float(pt[0]), 1), round(float(pt[1]), 1)] for pt in box_coords]

                is_confident = confidence >= self.config.confidence_threshold

                detections.append({
                    "text": raw_text,
                    "confidence": round(confidence, 4),
                    "bounding_box": rounded_box,
                    "box_flat": flat_coords,
                    "is_confident": is_confident
                })
            except Exception as item_err:
                logger.warning(f"Error parsing detection item: {item_err}, raw item: {line}")
                continue

        result_payload["detections"] = detections
        logger.debug(f"OCR extracted {len(detections)} lines for {image_name} [{variant_name}]")
        return result_payload
