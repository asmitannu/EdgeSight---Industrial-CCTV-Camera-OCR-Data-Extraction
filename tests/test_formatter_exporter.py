"""
Tests for DataFormatter and DataExporter modules.
"""

import os
import tempfile
import pytest
import pandas as pd

from src.data_formatter import DataFormatter
from src.exporter import DataExporter


@pytest.fixture
def sample_ocr_result():
    return {
        "image_name": "equipment_001.jpg",
        "variant_name": "enhanced_clahe",
        "detections": [
            {
                "text": "MOTOR-ABC-123",
                "confidence": 0.98,
                "bounding_box": [[10, 10], [100, 10], [100, 40], [10, 40]]
            },
            {
                "text": "VOLTAGE 400V",
                "confidence": 0.95,
                "bounding_box": [[10, 50], [120, 50], [120, 80], [10, 80]]
            }
        ]
    }


def test_calculate_alphanumeric_density():
    assert DataFormatter.calculate_alphanumeric_density("ABC123") == 1.0
    assert DataFormatter.calculate_alphanumeric_density("ABC-123/01") > 0.7
    assert DataFormatter.calculate_alphanumeric_density("!@#$%^") == 0.0
    assert DataFormatter.calculate_alphanumeric_density("") == 0.0


def test_score_variant(sample_ocr_result):
    score = DataFormatter.score_variant(sample_ocr_result)
    assert score > 0.5


def test_select_best_variant(sample_ocr_result):
    low_quality_result = {
        "image_name": "equipment_001.jpg",
        "variant_name": "noisy",
        "detections": [
            {
                "text": "...",
                "confidence": 0.20,
                "bounding_box": [[0, 0], [10, 0], [10, 10], [0, 10]]
            }
        ]
    }

    variants = {
        "enhanced_clahe": sample_ocr_result,
        "noisy": low_quality_result
    }

    best = DataFormatter.select_best_variant(variants)
    assert best["variant_name"] == "enhanced_clahe"


def test_to_structured_records(sample_ocr_result):
    records = DataFormatter.to_structured_records(sample_ocr_result, timestamp="2026-08-25T10:00:00Z")
    assert len(records) == 2
    assert records[0]["Image Name"] == "equipment_001.jpg"
    assert records[0]["Detected Text"] == "MOTOR-ABC-123"
    assert records[0]["Confidence"] == 0.98
    assert records[0]["Processing Variant"] == "enhanced_clahe"
    assert records[0]["Timestamp"] == "2026-08-25T10:00:00Z"


def test_export_csv_and_excel(sample_ocr_result):
    records = DataFormatter.to_structured_records(sample_ocr_result)

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "test_results.csv")
        xlsx_path = os.path.join(tmpdir, "test_results.xlsx")

        DataExporter.export_csv(records, csv_path)
        assert os.path.exists(csv_path)

        # Read back CSV to verify content
        df_csv = pd.read_csv(csv_path)
        assert len(df_csv) == 2
        assert list(df_csv.columns) == DataExporter.COLUMNS
        assert df_csv.iloc[0]["Detected Text"] == "MOTOR-ABC-123"

        DataExporter.export_excel(records, xlsx_path)
        assert os.path.exists(xlsx_path)

        # Read back Excel to verify content
        df_xlsx = pd.read_excel(xlsx_path)
        assert len(df_xlsx) == 2
        assert df_xlsx.iloc[0]["Detected Text"] == "MOTOR-ABC-123"
