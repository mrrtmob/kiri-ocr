"""
DB (Differentiable Binarization) text detector module.

Uses OpenCV's DNN module with pre-trained DB models for text detection.
"""
from .model import DBDetector

__all__ = ['DBDetector']
