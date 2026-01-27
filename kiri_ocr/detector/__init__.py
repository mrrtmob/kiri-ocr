from typing import List, Tuple, Union, Optional, Dict
from pathlib import Path
import os
import warnings
import numpy as np

# Import base components
from .base import TextBox, DetectionLevel

# Import legacy components
from .legacy import ImageProcessingTextDetector

# Import CRAFT detector
try:
    from .craft import CRAFTDetector
except ImportError:
    CRAFTDetector = None

# Import DB detector
try:
    from .db import DBDetector
except ImportError:
    DBDetector = None

# Legacy YOLO support (removed)
YOLOTextDetector = None

class TextDetector:
    """
    Unified Text Detector that supports CRAFT-based, DB-based and classic computer vision approaches.
    Defaults to CRAFT if available, otherwise falls back to classic.
    """
    
    def __init__(self, method: str = 'craft', model_path: Optional[str] = None, **kwargs):
        """
        Initialize the TextDetector.
        
        Args:
            method: 'craft', 'db', or 'legacy' (classic CV)
            model_path: Path to model (optional)
            **kwargs: Arguments passed to detector
        """
        self.conf_threshold = kwargs.pop('conf_threshold', 0.25)
        self.method = method
        self.kwargs = kwargs
        self.craft_detector = None
        self.db_detector = None
        
        # Try to resolve model path if not provided
        if model_path is None:
             possible_paths = []
             if self.method == 'db':
                 possible_paths = [
                     # Local project paths
                     'models/detector.onnx',
                     'detector.onnx',
                     # Legacy model names
                     'DB_TD500_resnet50.onnx',
                     'DB_IC15_resnet18.onnx',
                     # Package directory
                     os.path.join(os.path.dirname(__file__), 'detector.onnx'),
                     os.path.join(os.path.dirname(__file__), 'db', 'detector.onnx'),
                     os.path.join(os.path.dirname(__file__), '..', 'models', 'detector.onnx'),
                 ]
             else: # CRAFT
                 possible_paths = [
                     'runs/detect/khmer_text_detector/weights/best.pth',
                     'best.pth',  # Check current directory
                     os.path.join(os.path.dirname(__file__), 'best.pth'), # Check package directory
                 ]
                 
             for p in possible_paths:
                 if os.path.exists(p):
                     model_path = p
                     break
            
             # If still no model path and method requires one, default to official repo
             if model_path is None and self.method in ['db', 'craft']:
                 model_path = "mrrtmob/kiri-ocr"

        # Resolve HuggingFace path
        if model_path and "/" in model_path and not os.path.exists(model_path) and not model_path.startswith((".", "/")):
             try:
                 from huggingface_hub import hf_hub_download
                 hf_path = None
                 # Check for detector model in specific subfolder
                 # Prefer detector/DB/detector.onnx as requested
                 try:
                     hf_path = hf_hub_download(repo_id=model_path, filename="detector/DB/detector.onnx")
                 except Exception:
                     try:
                         # Fallback to detector/detector.onnx
                         hf_path = hf_hub_download(repo_id=model_path, filename="detector/detector.onnx")
                     except Exception:
                         pass
                 
                 if hf_path and os.path.exists(hf_path):
                     model_path = hf_path
                 else:
                     print(f"Warning: Could not find detector model in HuggingFace repo: {model_path}")
                     model_path = None
             except ImportError:
                 print("Warning: huggingface_hub not installed. Install with: pip install huggingface_hub")
                 model_path = None
             except Exception as e:
                 print(f"Warning: Could not download detector from HuggingFace: {e}")
                 model_path = None
        
        self.model_path = model_path
        
        # Initialize Detector based on method
        if self.method == 'craft':
             if CRAFTDetector is None:
                 warnings.warn("CRAFT detector not available. Falling back to legacy.")
                 self.method = 'legacy'
             else:
                 try:
                     self.craft_detector = CRAFTDetector()
                     # Load weights
                     if self.model_path and os.path.exists(self.model_path):
                         self.craft_detector.load_weights(self.model_path)
                     elif self.model_path:
                         print(f"Warning: CRAFT model path not found: {self.model_path}")
                 except Exception as e:
                     print(f"Error loading CRAFT detector: {e}. Falling back to legacy.")
                     self.method = 'legacy'
                     
        elif self.method == 'db':
             if DBDetector is None:
                 warnings.warn("DB detector not available. Falling back to legacy.")
                 self.method = 'legacy'
             else:
                 try:
                     if self.model_path and os.path.exists(self.model_path):
                         # Extract DB specific kwargs
                         db_kwargs = {k: v for k, v in self.kwargs.items() if k in [
                             # New parameters
                             'use_gpu', 'det_db_thresh', 'det_db_box_thresh',
                             'det_db_unclip_ratio', 'max_side_len', 'min_size',
                             # Legacy parameter aliases (backward compatibility)
                             'input_size', 'binary_threshold', 'polygon_threshold',
                             'max_candidates', 'unclip_ratio', 'padding_pct', 'padding_px',
                             'padding_y_pct', 'padding_y_px', 'line_tolerance_ratio', 'debug'
                         ]}
                         self.db_detector = DBDetector(self.model_path, **db_kwargs)
                     else:
                         print(f"Warning: DB model path not found: {self.model_path}")
                         self.method = 'legacy'
                 except Exception as e:
                     print(f"Error loading DB detector: {e}. Falling back to legacy.")
                     self.method = 'legacy'
        
        # Always initialize legacy detector as fallback/helper
        self.legacy_detector = ImageProcessingTextDetector(**kwargs)

    def detect_lines(self, image: Union[str, Path, np.ndarray]) -> List[Tuple[int, int, int, int]]:
        """Detect text lines. Returns bboxes only."""
        boxes = self.detect_lines_objects(image)
        return [b.bbox for b in boxes]

    def detect_lines_objects(self, image: Union[str, Path, np.ndarray]) -> List[TextBox]:
        """Detect text lines. Returns TextBox objects (bbox + confidence)."""
        if self.method == 'craft' and self.craft_detector:
            try:
                # CRAFT returns list of boxes (x1, y1, x2, y2)
                # We need to handle str/Path vs ndarray for detect_text
                if isinstance(image, (str, Path)):
                    img_path = str(image)
                    detected_boxes = self.craft_detector.detect_text(img_path)
                else:
                    # TODO: Support direct numpy array in CRAFTDetector.detect_text
                    # For now, fallback or temporary save
                    warnings.warn("CRAFT detector currently requires file path. Falling back to legacy.")
                    # Legacy fallback
                    return self._wrap_legacy(self.legacy_detector.detect_lines(image))
                
                return self._process_boxes_objects(detected_boxes, merge=True)
                
            except Exception as e:
                print(f"CRAFT detection failed: {e}. Falling back to legacy.")
                return self._wrap_legacy(self.legacy_detector.detect_lines(image))

        elif self.method == 'db' and self.db_detector:
            try:
                detected_boxes = self.db_detector.detect_text(image)
                # DB detector already returns sorted boxes, skip re-sorting
                return self._process_boxes_objects(detected_boxes, merge=False, skip_sort=True)
            except Exception as e:
                print(f"DB detection failed: {e}. Falling back to legacy.")
                return self._wrap_legacy(self.legacy_detector.detect_lines(image))
        
        return self._wrap_legacy(self.legacy_detector.detect_lines(image))

    def _wrap_legacy(self, bboxes):
        return [TextBox(x, y, w, h, confidence=1.0, level=DetectionLevel.LINE) for (x, y, w, h) in bboxes]

    def _process_boxes(self, detected_boxes, merge=True):
        """Convert polygons to TextBoxes (returns bboxes)."""
        boxes = self._process_boxes_objects(detected_boxes, merge=merge)
        return [b.bbox for b in boxes]

    def _process_boxes_objects(self, detected_boxes, merge=True, skip_sort=False) -> List[TextBox]:
        """Convert polygons to TextBoxes (returns objects)."""
        boxes = []
        padding = self.kwargs.get('padding', 0)
        
        for item in detected_boxes:
            # Handle (box, confidence) tuple from DB
            if isinstance(item, tuple) and len(item) == 2:
                 # Check if second item is float-like (confidence)
                 box, confidence = item
            else:
                 box = item
                 confidence = 1.0

            # Handle polygon output (4, 2)
            if hasattr(box, 'shape') and box.shape == (4, 2):
                x_min = np.min(box[:, 0])
                y_min = np.min(box[:, 1])
                x_max = np.max(box[:, 0])
                y_max = np.max(box[:, 1])
                x1, y1, x2, y2 = x_min, y_min, x_max, y_max
            else:
                x1, y1, x2, y2 = box
                
            w = x2 - x1
            h = y2 - y1
            
            # Apply padding
            if padding:
                    x1 = max(0, x1 - padding)
                    y1 = max(0, y1 - padding)
                    w += 2 * padding
                    h += 2 * padding
            
            boxes.append(TextBox(int(x1), int(y1), int(w), int(h), confidence=float(confidence), level=DetectionLevel.LINE))
        
        # Only sort if not already sorted (e.g., DB detector already sorts)
        if not skip_sort:
            boxes = self._sort_reading_order(boxes)
        if merge:
            boxes = self._merge_overlapping_boxes(boxes)
        return boxes

    def _sort_reading_order(self, boxes: List[TextBox]) -> List[TextBox]:
        """Sort boxes from top-to-bottom, left-to-right using robust line grouping."""
        if not boxes:
            return []
            
        # Helper to get center y
        def get_cy(b): return b.y + b.height / 2
        def get_cx(b): return b.x + b.width / 2
        
        # Sort initially by center Y
        boxes.sort(key=get_cy)
        
        # Calculate dynamic tolerance based on median height (more robust than average)
        heights = [b.height for b in boxes]
        median_h = float(np.median(heights)) if heights else 20.0
        # Use 70% of median height as tolerance for line grouping
        y_tolerance = median_h * 0.7
        
        # Group into lines
        lines = []
        current_line = [boxes[0]]
        
        for b in boxes[1:]:
            cy = get_cy(b)
            # Calculate the average center Y of the current line
            avg_line_cy = np.mean([get_cy(lb) for lb in current_line])
            
            if abs(cy - avg_line_cy) < y_tolerance:
                # Same line - add to current line
                current_line.append(b)
            else:
                # New line - save current and start new
                lines.append(current_line)
                current_line = [b]
                
        # Don't forget the last line
        if current_line:
            lines.append(current_line)
        
        # Sort each line by X (left to right) and flatten
        sorted_boxes = []
        for line in lines:
            # Sort boxes in this line by center X (left to right)
            line.sort(key=get_cx)
            sorted_boxes.extend(line)
        
        return sorted_boxes

    def detect_words(self, image: Union[str, Path, np.ndarray]) -> List[Tuple[int, int, int, int]]:
        """Detect words."""
        # YOLO typically detects lines. For words, we fallback to legacy
        # or implement segmentation on YOLO lines (future work).
        return self.legacy_detector.detect_words(image)
        
    def detect_blocks(self, image: Union[str, Path, np.ndarray]) -> List[Tuple[int, int, int, int]]:
        """Detect text blocks."""
        if (self.method == 'craft' and self.craft_detector) or (self.method == 'db' and self.db_detector):
             # Get lines using CRAFT or DB
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

