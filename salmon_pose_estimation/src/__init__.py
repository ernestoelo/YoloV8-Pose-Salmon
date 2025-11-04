# src/__init__.py
"""
Salmon Pose Estimation Package
"""
__version__ = "1.0.0"

from .metrics import PoseEvaluator, PCKMetric, OKSMetric
from .callbacks import CustomMetricsCallback
from .models import YOLOv8PoseTrainer
from .utils import ModelDownloader

__all__ = [
    'PoseEvaluator',
    'PCKMetric',
    'OKSMetric',
    'CustomMetricsCallback',
    'YOLOv8PoseTrainer',
    'ModelDownloader',
]
