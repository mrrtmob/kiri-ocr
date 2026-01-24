"""
CRAFT (Character Region Awareness for Text detection) detector module.
"""
from .model import CRAFT, CRAFTDetector
from . import utils
from . import imgproc

__all__ = ['CRAFT', 'CRAFTDetector', 'utils', 'imgproc']
