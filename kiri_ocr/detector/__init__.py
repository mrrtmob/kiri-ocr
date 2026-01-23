from typing import List, Tuple, Union, Optional, Dict
from pathlib import Path
import os
import warnings
import numpy as np

# Import legacy components
from .legacy import (
    ImageProcessingTextDetector, 
    TextBox, 
    DetectionLevel
)

# Import YOLO detector
try:
    from .yolo import YOLOTextDetector
except ImportError:
    YOLOTextDetector = None

class TextDetector:
    """
    Unified Text Detector that supports both YOLO-based and classic computer vision approaches.
    Defaults to YOLO if available, otherwise falls back to classic.
    """
    
    def __init__(self, method: str = 'yolo', model_path: Optional[str] = None, **kwargs):
        """
        Initialize the TextDetector.
        
        Args:
            method: 'yolo' or 'legacy' (classic CV)
            model_path: Path to YOLO model (optional)
            **kwargs: Arguments passed to ImageProcessingTextDetector
        """
        self.conf_threshold = kwargs.pop('conf_threshold', 0.25)
        self.method = method
        self.kwargs = kwargs
        self.yolo_detector = None
        
        # Try to resolve model path if not provided
        if model_path is None:
             possible_paths = [
                 'runs/detect/khmer_text_detector/weights/best.pt',
                 'best.pt',  # Check current directory
                 os.path.join(os.path.dirname(__file__), 'best.pt'), # Check package directory
             ]
             for p in possible_paths:
                 if os.path.exists(p):
                     model_path = p
                     break
        
        self.model_path = model_path
        
        # Initialize YOLO if requested
        if self.method == 'yolo':
             if YOLOTextDetector is None:
                 warnings.warn("YOLO detector not available (ultralytics not installed?). Falling back to legacy.")
                 self.method = 'legacy'
             elif self.model_path and os.path.exists(self.model_path):
                 try:
                     self.yolo_detector = YOLOTextDetector(self.model_path)
                     # print(f"Loaded YOLO detector from {self.model_path}")
                 except Exception as e:
                     print(f"Error loading YOLO detector: {e}. Falling back to legacy.")
                     self.method = 'legacy'
             else:
                 # Silently fallback if model not found, as user might not have trained it yet
                 # warnings.warn(f"YOLO model not found at {self.model_path}. Using legacy detector.")
                 self.method = 'legacy'
        
        # Always initialize legacy detector as fallback/helper
        self.legacy_detector = ImageProcessingTextDetector(**kwargs)

    def detect_lines(self, image: Union[str, Path, np.ndarray]) -> List[Tuple[int, int, int, int]]:
        """Detect text lines."""
        if self.method == 'yolo' and self.yolo_detector:
            try:
                detections = self.yolo_detector.detect_image(image, conf_threshold=self.conf_threshold, show_labels=False)
                boxes = []
                padding = self.kwargs.get('padding', 0)
                
                for d in detections:
                    x, y, x2, y2 = d['bbox']
                    conf = d['confidence']
                    w = x2 - x
                    h = y2 - y
                    
                    # Apply padding
                    if padding:
                         x = max(0, x - padding)
                         y = max(0, y - padding)
                         w += 2 * padding
                         h += 2 * padding
                    
                    boxes.append(TextBox(x, y, w, h, confidence=conf, level=DetectionLevel.LINE))
                
                boxes = sorted(boxes, key=lambda b: b.y)
                boxes = self._merge_overlapping_boxes(boxes)
                return [b.bbox for b in boxes]
                
            except Exception as e:
                print(f"YOLO detection failed: {e}. Falling back to legacy.")
                return self.legacy_detector.detect_lines(image)
        
        return self.legacy_detector.detect_lines(image)

    def detect_words(self, image: Union[str, Path, np.ndarray]) -> List[Tuple[int, int, int, int]]:
        """Detect words."""
        # YOLO typically detects lines. For words, we fallback to legacy
        # or implement segmentation on YOLO lines (future work).
        return self.legacy_detector.detect_words(image)
        
    def detect_blocks(self, image: Union[str, Path, np.ndarray]) -> List[Tuple[int, int, int, int]]:
        """Detect text blocks."""
        if self.method == 'yolo' and self.yolo_detector:
             # Get lines using YOLO
             lines_bbox = self.detect_lines(image)
             
             # Convert back to TextBox for grouping
             lines = [
                 TextBox(x, y, w, h, level=DetectionLevel.LINE) 
                 for (x, y, w, h) in lines_bbox
             ]
             
             # We need image dimensions for grouping
             img = self.legacy_detector._load_image(image)
             if img is None: 
                 return []
             h, w = img.shape[:2]
             
             # Use legacy grouping logic
             blocks = self.legacy_detector._group_lines_into_blocks(lines, w, h)
             return [b.bbox for b in blocks]
             
        return self.legacy_detector.detect_blocks(image)
        
    def detect_characters(self, image: Union[str, Path, np.ndarray]) -> List[Tuple[int, int, int, int]]:
        """Detect characters."""
        return self.legacy_detector.detect_characters(image)
        
    def detect_all(self, image: Union[str, Path, np.ndarray]) -> List[TextBox]:
        """Detect full hierarchy."""
        return self.legacy_detector.detect_all(image)

    def _merge_overlapping_boxes(self, boxes: List[TextBox], iou_threshold: float = 0.3) -> List[TextBox]:
        """Merge boxes with high vertical overlap (similar to test_yolo.py logic)."""
        if not boxes:
            return []
        
        # Sort by y
        boxes = sorted(boxes, key=lambda b: b.y)
        merged = []
        current = boxes[0]
        
        for next_box in boxes[1:]:
            # Calculate vertical overlap
            y1_curr, y2_curr = current.y, current.y + current.height
            y1_next, y2_next = next_box.y, next_box.y + next_box.height
            
            overlap_y = max(0, min(y2_curr, y2_next) - max(y1_curr, y1_next))
            min_h = min(current.height, next_box.height)
            
            overlap_ratio = overlap_y / min_h if min_h > 0 else 0
            
            if overlap_ratio > iou_threshold:
                # Merge
                x1 = min(current.x, next_box.x)
                y1 = min(current.y, next_box.y)
                x2 = max(current.x + current.width, next_box.x + next_box.width)
                y2 = max(current.y + current.height, next_box.y + next_box.height)
                
                # Average confidence
                conf = (current.confidence + next_box.confidence) / 2
                
                current = TextBox(x1, y1, x2 - x1, y2 - y1, confidence=conf, level=current.level)
            else:
                merged.append(current)
                current = next_box
                
        merged.append(current)
        return merged

    def is_multiline(self, image: Union[str, Path, np.ndarray], threshold: int = 2) -> bool:
        """Check if image contains multiple text lines."""
        lines = self.detect_lines(image)
        return len(lines) >= threshold
    
    def get_debug_images(self) -> Dict[str, np.ndarray]:
        """Get debug images."""
        return self.legacy_detector.get_debug_images()


# ==================== CONVENIENCE FUNCTIONS ====================

def detect_text_lines(image: Union[str, Path, np.ndarray], **kwargs) -> List[Tuple[int, int, int, int]]:
    """Convenience function to detect text lines."""
    detector = TextDetector(**kwargs)
    return detector.detect_lines(image)


def detect_text_words(image: Union[str, Path, np.ndarray], **kwargs) -> List[Tuple[int, int, int, int]]:
    """Convenience function to detect text words."""
    detector = TextDetector(**kwargs)
    return detector.detect_words(image)


def detect_text_blocks(image: Union[str, Path, np.ndarray], **kwargs) -> List[Tuple[int, int, int, int]]:
    """Convenience function to detect text blocks."""
    detector = TextDetector(**kwargs)
    return detector.detect_blocks(image)
