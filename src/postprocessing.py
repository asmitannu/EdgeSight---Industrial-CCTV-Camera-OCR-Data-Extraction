"""
OCR text postprocessing module.
Cleans and normalizes extracted text while strictly preserving industrial serial numbers,
equipment model IDs, ratings, timestamps, and numerical data.
"""

import re
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class TextPostprocessor:
    """Provides modular sanitization and normalization for OCR text outputs."""

    # Regex patterns for normalization
    MULTIPLE_SPACES_RE = re.compile(r"[ \t]+")
    CONTROL_CHARS_RE = re.compile(r"[\x00-\x1f\x7f-\x9f]")
    LINE_BREAKS_RE = re.compile(r"[\r\n]+")

    @classmethod
    def clean_text(cls, text: str) -> str:
        """
        Sanitizes and normalizes OCR text.

        - Removes non-printable control characters
        - Normalizes multiple spaces into a single space
        - Strips leading and trailing whitespace
        - Preserves alphanumeric IDs, hyphens, slashes, decimal points, colons, units

        Args:
            text: Raw extracted OCR string.

        Returns:
            str: Cleaned text.
        """
        if not text or not isinstance(text, str):
            return ""

        # Remove control characters
        cleaned = cls.CONTROL_CHARS_RE.sub("", text)

        # Replace internal newlines with space
        cleaned = cls.LINE_BREAKS_RE.sub(" ", cleaned)

        # Collapse multiple consecutive whitespace characters
        cleaned = cls.MULTIPLE_SPACES_RE.sub(" ", cleaned)

        # Trim edges
        cleaned = cleaned.strip()

        return cleaned

    @classmethod
    def postprocess_detections(
        cls,
        ocr_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Postprocesses all detection items in an OCR result dictionary.

        Args:
            ocr_result: Dict with 'image_name', 'variant_name', 'detections'.

        Returns:
            Dict[str, Any]: Processed OCR result with cleaned texts.
        """
        processed_result = {
            "image_name": ocr_result.get("image_name", "unknown"),
            "variant_name": ocr_result.get("variant_name", "original"),
            "detections": []
        }

        for item in ocr_result.get("detections", []):
            raw_text = item.get("text", "")
            cleaned_text = cls.clean_text(raw_text)

            # Skip detections that became completely empty
            if not cleaned_text:
                continue

            cleaned_item = dict(item)
            cleaned_item["text"] = cleaned_text
            processed_result["detections"].append(cleaned_item)

        return processed_result
