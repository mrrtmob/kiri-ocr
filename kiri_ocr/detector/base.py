"""
Base classes for text detection.
Contains common data structures shared across all detector modules.
"""
from dataclasses import dataclass, field
from typing import List, Tuple
from enum import Enum


class DetectionLevel(Enum):
    """Detection granularity levels"""
    BLOCK = "block"
    PARAGRAPH = "paragraph"
    LINE = "line"
    WORD = "word"
    CHARACTER = "character"


@dataclass
class TextBox:
    """Represents a detected text region with metadata"""
    x: int
    y: int
    width: int
    height: int
    confidence: float = 1.0
    level: DetectionLevel = DetectionLevel.LINE
    children: List['TextBox'] = field(default_factory=list)
    
    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        """Return (x, y, w, h) tuple"""
        return (self.x, self.y, self.width, self.height)
    
    @property
    def xyxy(self) -> Tuple[int, int, int, int]:
        """Return (x1, y1, x2, y2) tuple"""
        return (self.x, self.y, self.x + self.width, self.y + self.height)
    
    @property
    def area(self) -> int:
        return self.width * self.height
    
    @property
    def center(self) -> Tuple[float, float]:
        return (self.x + self.width / 2, self.y + self.height / 2)
    
    @property
    def baseline_y(self) -> float:
        """Approximate baseline (bottom - 20% of height)"""
        return self.y + self.height * 0.8
    
    def __repr__(self):
        return f"TextBox({self.x}, {self.y}, {self.width}, {self.height}, conf={self.confidence:.2f})"
