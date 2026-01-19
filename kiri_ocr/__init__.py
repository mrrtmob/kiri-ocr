from .core import OCR
from .renderer import DocumentRenderer
from .model import LightweightOCR, CharacterSet
from .detector import TextDetector

__version__ = '0.1.5'

__all__ = [
    'OCR',
    'DocumentRenderer',
    'LightweightOCR',
    'CharacterSet',
    'TextDetector',
]
