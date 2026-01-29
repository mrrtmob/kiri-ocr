"""
DB (Differentiable Binarization) Text Detector.

This module implements text detection using the DB (Differentiable Binarization) method
"""
import cv2
import numpy as np
from typing import List, Union, Tuple, Optional
from pathlib import Path

try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False

try:
    import pyclipper
    from shapely.geometry import Polygon
    HAS_POLYGON = True
except ImportError:
    HAS_POLYGON = False


class DBDetector:
    """
    DB (Differentiable Binarization) text detector using ONNX Runtime.
    
    Supports both CPU and GPU inference via ONNX Runtime.
    
    Example:
        detector = DBDetector('models/en_PP-OCRv3_det.onnx')
        results = detector.detect_text('image.jpg')
        for box, confidence in results:
            print(f"Box: {box}, Confidence: {confidence}")
    """
    
    def __init__(
        self,
        model_path: str,
        use_gpu: bool = False,
        # DB algorithm parameters
        det_db_thresh: float = 0.3,
        det_db_box_thresh: float = 0.5,
        det_db_unclip_ratio: float = 1.6,
        max_side_len: int = 960,
        min_size: int = 3,
        # Legacy parameter aliases (for backward compatibility)
        binary_threshold: Optional[float] = None,
        polygon_threshold: Optional[float] = None,
        unclip_ratio: Optional[float] = None,
        input_size: Optional[Tuple[int, int]] = None,
        max_candidates: int = 1000,
        # Padding parameters (for compatibility)
        padding_pct: float = 0.01,
        padding_px: int = 5,
        padding_y_pct: float = 0.05,
        padding_y_px: int = 5,
        line_tolerance_ratio: float = 0.7,
        # Debug mode
        debug: bool = False,
    ):
        """
        Initialize DB text detector.

        Args:
            model_path: Path to ONNX model
            use_gpu: Whether to use GPU for inference
            det_db_thresh: Threshold for binarization (pixels above this become foreground)
            det_db_box_thresh: Threshold for box confidence (boxes below this are filtered)
            det_db_unclip_ratio: Ratio for expanding detected boxes
            max_side_len: Maximum side length for input image
            min_size: Minimum box size to keep
            binary_threshold: Legacy alias for det_db_thresh
            polygon_threshold: Legacy alias for det_db_box_thresh
            unclip_ratio: Legacy alias for det_db_unclip_ratio
            input_size: Legacy parameter (ignored, use max_side_len instead)
            max_candidates: Maximum number of contour candidates to process
            padding_pct: Horizontal padding percentage for boxes
            padding_px: Horizontal padding in pixels for boxes
            padding_y_pct: Vertical padding percentage for boxes
            padding_y_px: Vertical padding in pixels for boxes
            line_tolerance_ratio: Ratio for line grouping tolerance
            debug: Enable debug output
        """
        if not HAS_ORT:
            raise ImportError("onnxruntime is required for DBDetector. Install with: pip install onnxruntime")
        
        if not HAS_POLYGON:
            raise ImportError("pyclipper and shapely are required for DBDetector. Install with: pip install pyclipper shapely")
        
        # Handle legacy parameter aliases
        self.det_db_thresh = binary_threshold if binary_threshold is not None else det_db_thresh
        self.det_db_box_thresh = polygon_threshold if polygon_threshold is not None else det_db_box_thresh
        self.det_db_unclip_ratio = unclip_ratio if unclip_ratio is not None else det_db_unclip_ratio
        self.max_side_len = max_side_len
        self.min_size = min_size
        self.max_candidates = max_candidates
        
        # Padding parameters
        self.padding_pct = padding_pct
        self.padding_px = padding_px
        self.padding_y_pct = padding_y_pct
        self.padding_y_px = padding_y_px
        self.line_tolerance_ratio = line_tolerance_ratio
        
        self.debug = debug
        self.model_path = model_path
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"DB model not found at {model_path}")
        
        # ONNX Runtime session
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_gpu else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        
        # Check input shape
        input_shape = self.session.get_inputs()[0].shape
        if self.debug:
            print(f"Model input name: {self.input_name}")
            print(f"Model input shape: {input_shape}")
        
        # Mean and std for normalization
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def _resize_image(self, img: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int], Tuple[float, float]]:
        """Resize image while maintaining aspect ratio."""
        h, w = img.shape[:2]
        
        # Calculate resize ratio
        ratio = 1.0
        if max(h, w) > self.max_side_len:
            ratio = self.max_side_len / max(h, w)
        
        new_h = int(h * ratio)
        new_w = int(w * ratio)
        
        # Make dimensions divisible by 32 (required by the model)
        new_h = max(32, int(round(new_h / 32) * 32))
        new_w = max(32, int(round(new_w / 32) * 32))
        
        resized = cv2.resize(img, (new_w, new_h))
        ratio_h = new_h / h
        ratio_w = new_w / w
        
        return resized, (h, w), (ratio_h, ratio_w)

    def _normalize(self, img: np.ndarray) -> np.ndarray:
        """Normalize image for model input."""
        # IMPORTANT: Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        img = img.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = img.transpose(2, 0, 1)  # HWC to CHW
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        return img.astype(np.float32)

    def _get_mini_boxes(self, contour: np.ndarray) -> Tuple[np.ndarray, float]:
        """Get minimum area bounding box."""
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])
        
        if points[1][1] > points[0][1]:
            index_1, index_4 = 0, 1
        else:
            index_1, index_4 = 1, 0
            
        if points[3][1] > points[2][1]:
            index_2, index_3 = 2, 3
        else:
            index_2, index_3 = 3, 2
            
        box = [points[index_1], points[index_2], points[index_3], points[index_4]]
        return np.array(box), min(bounding_box[1])

    def _box_score_fast(self, bitmap: np.ndarray, box: np.ndarray) -> float:
        """Calculate box score using the bitmap."""
        h, w = bitmap.shape[:2]
        box = box.copy()
        
        xmin = int(np.clip(np.floor(box[:, 0].min()), 0, w - 1))
        xmax = int(np.clip(np.ceil(box[:, 0].max()), 0, w - 1))
        ymin = int(np.clip(np.floor(box[:, 1].min()), 0, h - 1))
        ymax = int(np.clip(np.ceil(box[:, 1].max()), 0, h - 1))
        
        if xmax <= xmin or ymax <= ymin:
            return 0
        
        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
        
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]

    def _unclip(self, box: np.ndarray) -> Optional[np.ndarray]:
        """Expand the box using pyclipper."""
        poly = Polygon(box)
        if poly.area == 0 or poly.length == 0:
            return None
            
        distance = poly.area * self.det_db_unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box.astype(int).tolist(), pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = offset.Execute(distance)
        
        if len(expanded) == 0:
            return None
        return np.array(expanded[0])

    def _boxes_from_bitmap(
        self, 
        pred: np.ndarray, 
        bitmap: np.ndarray, 
        dest_width: int, 
        dest_height: int
    ) -> Tuple[List[np.ndarray], List[float]]:
        """Extract boxes from the prediction bitmap."""
        height, width = bitmap.shape
        
        # Find contours
        contours, _ = cv2.findContours(
            (bitmap * 255).astype(np.uint8), 
            cv2.RETR_LIST, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        boxes = []
        scores = []
        
        for contour in contours[:self.max_candidates]:
            if len(contour) < 4:
                continue
            
            # Get minimum bounding box
            box, sside = self._get_mini_boxes(contour)
            if sside < self.min_size:
                continue
            
            score = self._box_score_fast(pred, box)
            if score < self.det_db_box_thresh:
                continue
            
            # Expand the box
            expanded = self._unclip(box)
            if expanded is None:
                continue
                
            box, sside = self._get_mini_boxes(expanded)
            if sside < self.min_size + 2:
                continue
            
            # Scale back to original size
            box[:, 0] = np.clip(box[:, 0] / width * dest_width, 0, dest_width)
            box[:, 1] = np.clip(box[:, 1] / height * dest_height, 0, dest_height)
            
            boxes.append(box.astype(np.int32))
            scores.append(score)
        
        return boxes, scores

    def _sort_boxes_reading_order(
        self, results: List[Tuple[np.ndarray, float]]
    ) -> List[Tuple[np.ndarray, float]]:
        """
        Robust sorting for documents - groups boxes into lines based on vertical overlap.
        Uses bounding rectangle for consistent center calculation.
        """
        if not results:
            return []

        # 1. Calculate bounding rect centers and heights for all boxes
        box_data = []
        for box, conf in results:
            # Use simple bounding rectangle (more consistent than minAreaRect)
            x, y, w, h = cv2.boundingRect(box)
            center_x = x + w / 2
            center_y = y + h / 2
            
            box_data.append({
                "box": box,
                "conf": conf,
                "cy": center_y,
                "cx": center_x,
                "x": x,
                "h": h
            })

        # 2. Sort initially by vertical center (y)
        box_data.sort(key=lambda b: b['cy'])

        # 3. Calculate a dynamic tolerance based on the median height
        heights = [b['h'] for b in box_data]
        median_h = float(np.median(heights)) if heights else 20.0
        # Use 80% of median height as tolerance for line grouping
        y_tolerance = median_h * 0.8

        # 4. Group into lines
        lines = []
        current_line = []

        for item in box_data:
            if not current_line:
                current_line.append(item)
                continue

            # Calculate the average vertical center of the current line
            avg_line_y = np.mean([b['cy'] for b in current_line])

            # If this box's center is close enough to the line's average center
            if abs(item['cy'] - avg_line_y) < y_tolerance:
                current_line.append(item)
            else:
                # End of line, push to lines list and start new one
                lines.append(current_line)
                current_line = [item]

        # Append the last line
        if current_line:
            lines.append(current_line)

        # 5. Sort each line by X coordinate (left edge) and flatten
        sorted_results = []
        for line in lines:
            # Sort boxes in this line left-to-right by left edge (x), not center
            line.sort(key=lambda b: b['x'])
            for item in line:
                sorted_results.append((item['box'], item['conf']))

        return sorted_results

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
        max_pad_w = np.full(n, float("inf"))
        max_pad_h = np.full(n, float("inf"))

        for i in range(n):
            xi, yi, wi, hi = aabbs[i]

            for j in range(n):
                if i == j:
                    continue

                xj, yj, wj, hj = aabbs[j]

                # Check vertical band overlap (limits horizontal expansion)
                y_start = max(yi, yj)
                y_end = min(yi + hi, yj + hj)

                if y_start < y_end:  # Overlap in Y
                    dist_x = 0
                    if xi >= xj + wj:  # i is right of j
                        dist_x = xi - (xj + wj)
                    elif xj >= xi + wi:  # j is right of i
                        dist_x = xj - (xi + wi)
                    else:
                        dist_x = 0  # Overlap

                    max_pad_w[i] = min(max_pad_w[i], dist_x)

                # Check horizontal band overlap (limits vertical expansion)
                x_start = max(xi, xj)
                x_end = min(xi + wi, xj + wj)

                if x_start < x_end:  # Overlap in X
                    dist_y = 0
                    if yi >= yj + hj:  # i is below j
                        dist_y = yi - (yj + hj)
                    elif yj >= yi + hi:  # j is below i
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

    def detect(
        self, 
        img: np.ndarray, 
        return_scores: bool = False
    ) -> Union[List[np.ndarray], Tuple[List[np.ndarray], List[float]]]:
        """
        Detect text regions in the image.
        
        Args:
            img: Input image (BGR format, numpy array)
            return_scores: Whether to return confidence scores
            
        Returns:
            List of detected boxes, each box is 4 points [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            If return_scores=True, returns (boxes, scores) tuple
        """
        if img is None:
            return [] if not return_scores else ([], [])
        
        # Handle grayscale
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        # Handle RGBA
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        
        orig_h, orig_w = img.shape[:2]
        
        # Resize and normalize
        resized_img, (orig_h, orig_w), (ratio_h, ratio_w) = self._resize_image(img)
        input_tensor = self._normalize(resized_img)
        
        # Run inference
        outputs = self.session.run(None, {self.input_name: input_tensor})
        
        # Get probability map - handle different output formats
        pred = outputs[0]
        if len(pred.shape) == 4:
            pred = pred[0, 0]  # [batch, channel, h, w] -> [h, w]
        elif len(pred.shape) == 3:
            pred = pred[0]  # [batch, h, w] -> [h, w]
        
        if self.debug:
            print(f"  Pred shape: {pred.shape}, min: {pred.min():.4f}, max: {pred.max():.4f}")
        
        # Binarize
        bitmap = (pred > self.det_db_thresh).astype(np.float32)
        
        if self.debug:
            print(f"  Pixels above threshold: {bitmap.sum():.0f}")
        
        # Get boxes
        boxes, scores = self._boxes_from_bitmap(pred, bitmap, orig_w, orig_h)
        
        if return_scores:
            return boxes, scores
        return boxes

    def detect_text(
        self, image: Union[str, Path, np.ndarray]
    ) -> List[Tuple[np.ndarray, float]]:
        """
        Detect text in an image.
        
        This method provides compatibility with the TextDetector interface.
        
        Args:
            image: Path to image or numpy array (BGR format)
            
        Returns:
            List of (box, confidence) tuples, sorted in reading order
        """
        if isinstance(image, (str, Path)):
            image_cv = cv2.imread(str(image))
            if image_cv is None:
                raise ValueError(f"Image not found at {image}")
        elif isinstance(image, np.ndarray):
            image_cv = image.copy()
        else:
            raise TypeError("Image must be a path or numpy array")

        # Detect boxes with scores
        boxes, scores = self.detect(image_cv, return_scores=True)
        
        if not boxes:
            return []
        
        # Apply smart padding
        padded_boxes = self._apply_smart_padding(boxes)
        
        # Combine with confidences
        results = list(zip(padded_boxes, scores))
        
        # Sort boxes in reading order (top-to-bottom, left-to-right)
        results = self._sort_boxes_reading_order(results)

        return results

    def __call__(self, img: np.ndarray) -> List[np.ndarray]:
        """Alias for detect method."""
        return self.detect(img)
