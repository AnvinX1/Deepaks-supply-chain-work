"""Detectors module."""
from .isolation_forest import IsolationForestDetector
from .autoencoder import AutoencoderDetector
from .statistical_methods import StatisticalDetector
__all__ = ["IsolationForestDetector", "AutoencoderDetector", "StatisticalDetector"]
