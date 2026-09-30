"""
Data formatting and multi-variant scoring module.
Evaluates OCR results from multiple preprocessing variants, selects the optimal variant,
and formats results into structured tabular records for CSV/Excel export.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple

from config.settings import VariantScoringConfig

logger = logging.getLogger(__name__)


class DataFormatter:
    """Calculates variant scores and formats OCR output into structured records."""

    @classmethod
    def calculate_alphanumeric_density(cls, text: str) -> float:
        """
        Calculates the ratio of alphanumeric characters to total characters.
        Useful to penalize random artifact noise (e.g. '~`^.,').
        """
        if not text:
            return 0.0
        alnum_count = sum(1 for c in text if c.isalnum())
        return alnum_count / max(len(text), 1)

    @classmethod
    def score_variant(
        cls,
        ocr_result: Dict[str, Any],
        scoring_config: Optional[VariantScoringConfig] = None
    ) -> float:
        """
        Calculates a quality score for an OCR variant output.

        Score components:
            1. Average confidence (0.0 to 1.0)
            2. Alphanumeric density (0.0 to 1.0)
            3. Detection quantity reward (normalized log scale)

        Args:
            ocr_result: OCR result dict with 'detections'.
            scoring_config: Scoring weights.

        Returns:
            float: Composite score (higher is better).
        """
        cfg = scoring_config or VariantScoringConfig()
        detections = ocr_result.get("detections", [])

        if not detections:
            return 0.0

        confidences = [d.get("confidence", 0.0) for d in detections]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        all_text = " ".join(d.get("text", "") for d in detections)
        alnum_density = cls.calculate_alphanumeric_density(all_text)

        # Word count bonus (normalized between 0.0 and 1.0, saturated at 10 words)
        word_count = len(all_text.split())
        word_score = min(word_count / 10.0, 1.0)

        composite_score = (
            cfg.confidence_weight * avg_confidence +
            cfg.alphanumeric_density_weight * alnum_density +
            cfg.word_count_weight * word_score
        )

        logger.debug(
            f"Variant '{ocr_result.get('variant_name')}' score: {composite_score:.3f} "
            f"(conf={avg_confidence:.2f}, alnum={alnum_density:.2f}, words={word_count})"
        )
        return composite_score

    @classmethod
    def select_best_variant(
        cls,
        variant_ocr_results: Dict[str, Dict[str, Any]],
        scoring_config: Optional[VariantScoringConfig] = None
    ) -> Dict[str, Any]:
        """
        Selects the best OCR result among multiple preprocessing variants.

        Args:
            variant_ocr_results: Dict of variant_name -> ocr_result.
            scoring_config: Scoring weights.

        Returns:
            Dict[str, Any]: The highest scoring OCR result.
        """
        if not variant_ocr_results:
            return {"image_name": "unknown", "variant_name": "none", "detections": []}

        scored_variants: List[Tuple[float, str, Dict[str, Any]]] = []

        for v_name, ocr_res in variant_ocr_results.items():
            score = cls.score_variant(ocr_res, scoring_config)
            scored_variants.append((score, v_name, ocr_res))

        # Sort descending by score
        scored_variants.sort(key=lambda item: item[0], reverse=True)

        best_score, best_variant_name, best_result = scored_variants[0]
        logger.info(
            f"Selected best variant '{best_variant_name}' (score: {best_score:.3f}, "
            f"detections: {len(best_result.get('detections', []))})"
        )

        return best_result

    @classmethod
    def to_structured_records(
        cls,
        ocr_result: Dict[str, Any],
        timestamp: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Transforms an OCR result into flat tabular records.

        Schema:
            - 'Image Name': str
            - 'Detected Text': str
            - 'Confidence': float
            - 'Bounding Box': str (JSON string representation)
            - 'Processing Variant': str
            - 'Timestamp': str (ISO 8601)

        Args:
            ocr_result: Standard OCR result dict.
            timestamp: Optional ISO string. If None, current UTC time is used.

        Returns:
            List[Dict[str, Any]]: List of structured rows.
        """
        image_name = ocr_result.get("image_name", "unknown")
        variant_name = ocr_result.get("variant_name", "original")
        detections = ocr_result.get("detections", [])

        current_time = timestamp or datetime.now(timezone.utc).isoformat()

        # If no detections, return an empty row indicating no text found
        if not detections:
            return [{
                "Image Name": image_name,
                "Detected Text": "",
                "Confidence": 0.0,
                "Bounding Box": "[]",
                "Processing Variant": variant_name,
                "Timestamp": current_time
            }]

        records = []
        for det in detections:
            bbox = det.get("bounding_box", [])
            records.append({
                "Image Name": image_name,
                "Detected Text": det.get("text", ""),
                "Confidence": round(float(det.get("confidence", 0.0)), 4),
                "Bounding Box": json.dumps(bbox),
                "Processing Variant": variant_name,
                "Timestamp": current_time
            })

        return records
