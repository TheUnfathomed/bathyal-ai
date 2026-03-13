"""Bathyal AI package."""

from .classifier import SpeciesClassifierBundle, ThresholdConfig
from .evaluation import EvaluationConfig, run_evaluation
from .pipeline import PipelineConfig, run_pipeline
from .training import TrainingConfig, train_classifier

__all__ = [
    "SpeciesClassifierBundle",
    "ThresholdConfig",
    "EvaluationConfig",
    "run_evaluation",
    "PipelineConfig",
    "run_pipeline",
    "TrainingConfig",
    "train_classifier",
]

try:
    from .vlm_classification import VlmClassificationClient
    from .vlm_detection import GeminiDetectionClient

    __all__ += ["GeminiDetectionClient", "VlmClassificationClient"]
except ImportError:
    pass
