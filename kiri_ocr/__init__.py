"""
Kiri OCR - Lightweight OCR for English and Khmer documents.

Main Components:
- OCR: Main OCR class for document processing
- KiriOCR: Transformer-based OCR model (CNN + Transformer encoder + CTC/Attention decoder)
- TextDetector: Text detection module
"""
from .core import OCR
from .renderer import DocumentRenderer
from .model import KiriOCR, CFG, CharTokenizer
from .detector import TextDetector

__version__ = '0.2.3'

__all__ = [
    'OCR',
    'DocumentRenderer',
    'KiriOCR',
    'CFG',
    'CharTokenizer',
    'TextDetector',
]
