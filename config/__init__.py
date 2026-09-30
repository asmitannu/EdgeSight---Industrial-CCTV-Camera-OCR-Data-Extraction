"""Configuration module for the Industrial OCR system."""
from config.settings import (
    PipelineConfig,
    OCRConfig,
    PreprocessingConfig,
    VariantScoringConfig,
    default_config,
)

__all__ = [
    "PipelineConfig",
    "OCRConfig",
    "PreprocessingConfig",
    "VariantScoringConfig",
    "default_config",
]
