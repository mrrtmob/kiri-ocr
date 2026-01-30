"""
Kiri OCR - Lightweight OCR for English and Khmer documents.

Main Components:
- OCR: Main OCR class for document processing
- KiriOCR: Transformer-based OCR model (CNN + Transformer encoder + CTC/Attention decoder)
- TextDetector: Text detection module
"""

__version__ = '0.2.12'

# Lazy imports for fast CLI startup
# Heavy modules (torch, cv2) are only loaded when actually used

def __getattr__(name):
    """Lazy import heavy modules only when accessed."""
    if name == 'OCR':
        from .core import OCR
        return OCR
    elif name == 'DocumentRenderer':
        from .renderer import DocumentRenderer
        return DocumentRenderer
    elif name == 'KiriOCR':
        from .model import KiriOCR
        return KiriOCR
    elif name == 'CFG':
        from .model import CFG
        return CFG
    elif name == 'CharTokenizer':
        from .model import CharTokenizer
        return CharTokenizer
    elif name == 'TextDetector':
        from .detector import TextDetector
        return TextDetector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    'OCR',
    'DocumentRenderer',
    'KiriOCR',
    'CFG',
    'CharTokenizer',
    'TextDetector',
]
