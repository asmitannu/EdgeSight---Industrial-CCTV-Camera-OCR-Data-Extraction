"""
Tests for TextPostprocessor module.
"""

import pytest
from src.postprocessing import TextPostprocessor


def test_clean_text_whitespace():
    raw = "  SIEMENS   SIMOTICS    3~  MOT  \n"
    cleaned = TextPostprocessor.clean_text(raw)
    assert cleaned == "SIEMENS SIMOTICS 3~ MOT"


def test_clean_text_preserves_alphanumeric_ids():
    test_cases = [
        ("MODEL:  1LE1001-1DB43-4AA4 ", "MODEL: 1LE1001-1DB43-4AA4"),
        ("SN-98420194/01", "SN-98420194/01"),
        ("400V / 690V  50Hz", "400V / 690V 50Hz"),
        ("IP55   EFF: 93.6%", "IP55 EFF: 93.6%"),
        ("1475 RPM", "1475 RPM"),
        ("TAG# MTR_01.A", "TAG# MTR_01.A")
    ]

    for raw, expected in test_cases:
        assert TextPostprocessor.clean_text(raw) == expected


def test_clean_text_control_characters():
    raw = "TEST\x00\x1fCODE\x7f"
    cleaned = TextPostprocessor.clean_text(raw)
    assert cleaned == "TESTCODE"


def test_postprocess_detections():
    ocr_result = {
        "image_name": "motor.jpg",
        "variant_name": "enhanced_clahe",
        "detections": [
            {
                "text": "  MODEL:   1LE1001  ",
                "confidence": 0.98,
                "bounding_box": [[0, 0], [10, 0], [10, 10], [0, 10]]
            },
            {
                "text": "   ",
                "confidence": 0.10,
                "bounding_box": [[0, 0], [10, 0], [10, 10], [0, 10]]
            }
        ]
    }

    processed = TextPostprocessor.postprocess_detections(ocr_result)
    assert len(processed["detections"]) == 1
    assert processed["detections"][0]["text"] == "MODEL: 1LE1001"
    assert processed["detections"][0]["confidence"] == 0.98
