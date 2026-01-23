import cv2
import numpy as np
import math
from typing import List, Union, Tuple
from pathlib import Path

class DBNetDetector:
    def __init__(self, model_path: str,
                 input_size: Tuple[int, int] = (1280, 736),
                 binary_threshold: float = 0.2,
                 polygon_threshold: float = 0.4,
                 max_candidates: int = 500,
                 unclip_ratio: float = 2.0,
                 padding_pct: float = 0.01,
                 padding_px: int = 4,
                 padding_y_pct: float = 0.01,
                 padding_y_px: int = 0,
                 mean: Tuple[float, float, float] = (122.67891434, 116.66876762, 104.00698793),
                 scale: float = 1.0 / 255.0):
        """
        DBNet text detector.
        
        Args:
            model_path: Path to ONNX model
            input_size: Input size for inference (width, height), must be multiples of 32
            binary_threshold: Threshold for binary map
            polygon_threshold: Threshold for polygons
            max_candidates: Max number of candidates
            unclip_ratio: Unclip ratio for expansion
            padding_pct: Horizontal expansion percentage (0.08 = 8%)
            padding_px: Horizontal expansion fixed pixels
            mean: Mean for normalization
            scale: Scale for normalization
        """
        
        self.model_path = model_path
        self.input_size = input_size
        self.binary_threshold = binary_threshold
        self.polygon_threshold = polygon_threshold
        self.max_candidates = max_candidates
        self.unclip_ratio = unclip_ratio
        self.padding_pct = padding_pct
        self.padding_px = padding_px
        self.padding_y_pct = padding_y_pct
        self.padding_y_px = padding_y_px
        self.mean = mean
        self.scale = scale
        
        if not Path(self.model_path).exists():
             raise FileNotFoundError(f"DBNet model not found at {self.model_path}")

        self.net = cv2.dnn.TextDetectionModel_DB(self.model_path)
        self.net.setInputParams(scale=self.scale, size=self.input_size, mean=self.mean, swapRB=True)
        self.net.setBinaryThreshold(self.binary_threshold)
        self.net.setPolygonThreshold(self.polygon_threshold)
        self.net.setMaxCandidates(self.max_candidates)
        self.net.setUnclipRatio(self.unclip_ratio)

    def detect_text(self, image: Union[str, Path, np.ndarray]) -> List[np.ndarray]:
        if isinstance(image, (str, Path)):
            image_cv = cv2.imread(str(image))
            if image_cv is None:
                raise ValueError(f"Image not found at {image}")
        elif isinstance(image, np.ndarray):
            image_cv = image
            if len(image_cv.shape) == 2:  # Grayscale
                image_cv = cv2.cvtColor(image_cv, cv2.COLOR_GRAY2BGR)
            elif len(image_cv.shape) == 3 and image_cv.shape[2] == 4: # RGBA
                image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGRA2BGR)
        else:
            raise TypeError("Image must be a path or numpy array")

        boxes, confidences = self.net.detect(image_cv)
        
        processed_boxes = []
        if boxes is not None:
            # Convert to numpy int32
            boxes_np = [np.int32(box) for box in boxes]
            processed_boxes = self._apply_smart_padding(boxes_np)
                
        return processed_boxes

    def _apply_smart_padding(self, boxes: List[np.ndarray]) -> List[np.ndarray]:
        """
        Apply padding dynamically to avoid overlaps.
        """
        if not boxes:
            return []
            
        n = len(boxes)
        # Calculate AABBs for distance checking
        aabbs = [cv2.boundingRect(b) for b in boxes]
        
        # Max allowable expansion (total width/height increase)
        max_pad_w = np.full(n, float('inf'))
        max_pad_h = np.full(n, float('inf'))
        
        for i in range(n):
            xi, yi, wi, hi = aabbs[i]
            
            for j in range(n):
                if i == j: continue
                
                xj, yj, wj, hj = aabbs[j]
                
                # Check vertical band overlap (limits horizontal expansion)
                # Determine vertical intersection
                y_start = max(yi, yj)
                y_end = min(yi + hi, yj + hj)
                
                if y_start < y_end: # Overlap in Y
                    # Calculate horizontal distance
                    dist_x = 0
                    if xi >= xj + wj: # i is right of j
                        dist_x = xi - (xj + wj)
                    elif xj >= xi + wi: # j is right of i
                        dist_x = xj - (xi + wi)
                    else:
                        dist_x = 0 # Overlap
                    
                    # Symmetric expansion means each side expands by dist_x/2
                    # Total width increase allowed is dist_x
                    max_pad_w[i] = min(max_pad_w[i], dist_x)
                    
                # Check horizontal band overlap (limits vertical expansion)
                x_start = max(xi, xj)
                x_end = min(xi + wi, xj + wj)
                
                if x_start < x_end: # Overlap in X
                    dist_y = 0
                    if yi >= yj + hj: # i is below j
                        dist_y = yi - (yj + hj)
                    elif yj >= yi + hi: # j is below i
                        dist_y = yj - (yi + hi)
                    else:
                        dist_y = 0
                    
                    max_pad_h[i] = min(max_pad_h[i], dist_y)

        final_boxes = []
        for i, box in enumerate(boxes):
            rect = cv2.minAreaRect(box)
            (center, (w, h), angle) = rect
            (cx, cy) = center
            
            # Ensure w is the "long" side
            if w < h:
                w, h = h, w
                angle += 90

            # Target padding
            # Use height (approx font size) to ensure minimum padding for short words
            # Add 0.5 * h to width (approx half char on each side)
            target_pad_w = (w * self.padding_pct) + (h * 0.5) + self.padding_px
            target_pad_h = (h * self.padding_y_pct) + self.padding_y_px
            
            # Clamp by neighbor limits
            actual_pad_w = min(target_pad_w, max(0, max_pad_w[i]))
            actual_pad_h = min(target_pad_h, max(0, max_pad_h[i]))
            
            new_w = w + actual_pad_w
            new_h = h + actual_pad_h
            
            new_rect = ((cx, cy), (new_w, new_h), angle)
            new_box = cv2.boxPoints(new_rect)
            final_boxes.append(np.int32(new_box))
            
        return final_boxes
