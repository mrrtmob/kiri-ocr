"""
Legacy (Classic Computer Vision) text detector module.

Uses traditional image processing techniques like:
- Multi-channel binarization
- MSER (Maximally Stable Extremal Regions)
- Gradient/Edge-based detection
- Connected Component Analysis
"""
from .detector import ImageProcessingTextDetector

__all__ = ['ImageProcessingTextDetector']
