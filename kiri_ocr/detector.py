import cv2
import numpy as np

class TextDetector:
    """Detect text regions robustly across all background/text color conditions"""
    
    def __init__(self, padding=None):
        """
        Args:
            padding: Pixels to add around detected boxes (default: None for auto)
                    If None, padding will be automatically calculated based on text size
        """
        self.padding = padding
        self._auto_padding = None
    
    def _preprocess_image(self, gray):
        """
        Enhanced preprocessing for robust text detection
        Returns multiple binary candidates
        """
        # Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        candidates = []
        
        # Method 1: Otsu's binarization on enhanced image
        _, binary_otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        candidates.append(('otsu', binary_otsu))
        
        # Method 2: Inverted Otsu
        candidates.append(('otsu_inv', 255 - binary_otsu))
        
        # Method 3: Adaptive threshold (good for uneven lighting)
        binary_adaptive = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 21, 10
        )
        candidates.append(('adaptive', binary_adaptive))
        
        # Method 4: Inverted adaptive
        candidates.append(('adaptive_inv', 255 - binary_adaptive))
        
        # Method 5: Sauvola-like local thresholding (better for varying backgrounds)
        binary_sauvola = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 31, 15
        )
        candidates.append(('sauvola', binary_sauvola))
        candidates.append(('sauvola_inv', 255 - binary_sauvola))
        
        return candidates
    
    def _select_best_binary(self, candidates, gray):
        """
        Select the best binarization by analyzing connected components
        Best = most text-like components with reasonable size distribution
        """
        best_score = -1
        best_binary = None
        best_stats = None
        
        for method_name, binary in candidates:
            # Get connected components
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                binary, connectivity=8
            )
            
            if num_labels <= 1:  # Only background
                continue
            
            # Analyze component statistics
            heights = []
            widths = []
            areas = []
            aspect_ratios = []
            
            for i in range(1, num_labels):
                x, y, w, h, area = stats[i]
                
                # Basic filters
                if w >= 2 and h >= 3 and area >= 6:
                    heights.append(h)
                    widths.append(w)
                    areas.append(area)
                    if h > 0:
                        aspect_ratios.append(w / h)
            
            if len(heights) < 3:  # Too few components
                continue
            
            # Calculate quality metrics
            median_h = np.median(heights)
            median_w = np.median(widths)
            std_h = np.std(heights)
            
            # Filter for text-like components
            text_like_count = 0
            for h, w, ar in zip(heights, widths, aspect_ratios):
                # Text characters typically have:
                # - Reasonable aspect ratio (not too extreme)
                # - Height within reasonable range of median
                if (0.1 < ar < 15 and 
                    median_h * 0.2 < h < median_h * 4 and
                    w > 2):
                    text_like_count += 1
            
            # Score this binarization
            # Prefer methods with:
            # 1. More text-like components
            # 2. Consistent height distribution (lower std relative to median)
            # 3. Reasonable median size (not too small/large)
            
            if median_h < 5 or median_h > gray.shape[0] * 0.5:
                continue  # Unrealistic text size
            
            height_consistency = 1.0 / (1.0 + std_h / max(median_h, 1))
            size_score = 1.0 if 8 <= median_h <= 200 else 0.5
            
            score = text_like_count * height_consistency * size_score
            
            if score > best_score:
                best_score = score
                best_binary = binary
                best_stats = {
                    'method': method_name,
                    'num_components': num_labels - 1,
                    'text_like': text_like_count,
                    'median_height': median_h,
                    'median_width': median_w,
                    'heights': heights,
                    'widths': widths
                }
        
        return best_binary, best_stats
    
    def detect_lines(self, image_path):
        """
        Detect text lines robustly across all background/text conditions
        
        Returns:
            List of bounding boxes [(x, y, w, h), ...]
        """
        # Load image
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        img_h, img_w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Get multiple binarization candidates
        candidates = self._preprocess_image(gray)
        
        # Select best binarization
        binary, stats = self._select_best_binary(candidates, gray)
        
        if binary is None or stats is None:
            # Fallback to simple Otsu
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            if np.mean(binary) > 127:
                binary = 255 - binary
        
        # Connected components analysis
        num_labels, labels, component_stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        
        # Collect and filter components
        heights = []
        widths = []
        valid_components = []
        
        for i in range(1, num_labels):
            x, y, w, h, area = component_stats[i]
            
            if w >= 2 and h >= 3 and area >= 6:
                heights.append(h)
                widths.append(w)
                valid_components.append({
                    'id': i,
                    'bbox': (x, y, w, h),
                    'area': area,
                    'centroid': centroids[i]
                })
        
        if not valid_components:
            return []
        
        # Estimate text size
        median_height = np.median(heights)
        median_width = np.median(widths)
        
        # Auto-calculate padding
        if self.padding is None:
            self._auto_padding = max(2, int(median_height * 0.15))
        else:
            self._auto_padding = self.padding
        
        # Filter components
        min_h = max(3, median_height * 0.25)
        max_h = median_height * 3.0
        min_w = max(2, median_width * 0.1)
        max_w = img_w * 0.95
        
        text_components = []
        for comp in valid_components:
            x, y, w, h = comp['bbox']
            
            if min_h <= h <= max_h and min_w <= w <= max_w:
                aspect = w / h if h > 0 else 0
                if 0.05 < aspect < 20:
                    if h >= median_height * 0.3:
                        text_components.append(comp)
        
        if not text_components:
            return []
        
        # Group into lines by baseline clustering
        text_components.sort(key=lambda c: c['centroid'][1])
        
        lines = []
        current_line = [text_components[0]]
        
        for comp in text_components[1:]:
            line_y_values = [c['centroid'][1] for c in current_line]
            line_y_center = np.mean(line_y_values)
            line_heights = [c['bbox'][3] for c in current_line]
            avg_line_height = np.mean(line_heights)
            
            comp_y = comp['centroid'][1]
            tolerance = avg_line_height * 0.45
            
            if abs(comp_y - line_y_center) <= tolerance:
                current_line.append(comp)
            else:
                lines.append(current_line)
                current_line = [comp]
        
        if current_line:
            lines.append(current_line)
        
        # Create bounding boxes
        line_boxes = []
        for line in lines:
            if not line:
                continue
            
            x_min = min(c['bbox'][0] for c in line)
            y_min = min(c['bbox'][1] for c in line)
            x_max = max(c['bbox'][0] + c['bbox'][2] for c in line)
            y_max = max(c['bbox'][1] + c['bbox'][3] for c in line)
            
            x_pad = max(0, x_min - self._auto_padding)
            y_pad = max(0, y_min - self._auto_padding)
            w_pad = min(img_w - x_pad, (x_max - x_min) + 2 * self._auto_padding)
            h_pad = min(img_h - y_pad, (y_max - y_min) + 2 * self._auto_padding)
            
            line_boxes.append((x_pad, y_pad, w_pad, h_pad))
        
        # Merge overlapping boxes
        line_boxes = self._merge_overlapping_boxes(line_boxes, median_height)
        
        # Sort top to bottom
        line_boxes = sorted(line_boxes, key=lambda b: b[1])
        
        return line_boxes
    
    def detect_words(self, image_path):
        """
        Detect words robustly across all conditions
        
        Returns:
            List of word bounding boxes
        """
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        img_h, img_w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Get best binarization
        candidates = self._preprocess_image(gray)
        binary, stats = self._select_best_binary(candidates, gray)
        
        if binary is None:
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            if np.mean(binary) > 127:
                binary = 255 - binary
        
        # Connected components
        num_labels, labels, component_stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        
        # Collect components
        heights = []
        widths = []
        valid_components = []
        
        for i in range(1, num_labels):
            x, y, w, h, area = component_stats[i]
            if w >= 2 and h >= 3 and area >= 6:
                heights.append(h)
                widths.append(w)
                valid_components.append({
                    'bbox': (x, y, w, h),
                    'centroid': centroids[i]
                })
        
        if not valid_components:
            return []
        
        median_height = np.median(heights)
        median_width = np.median(widths)
        
        if self.padding is None:
            self._auto_padding = max(2, int(median_height * 0.15))
        else:
            self._auto_padding = self.padding
        
        # Filter components
        min_h = max(3, median_height * 0.25)
        max_h = median_height * 3.0
        
        text_components = []
        for comp in valid_components:
            x, y, w, h = comp['bbox']
            if min_h <= h <= max_h and w < img_w * 0.95:
                aspect = w / h if h > 0 else 0
                if 0.05 < aspect < 20 and h >= median_height * 0.3:
                    text_components.append(comp)
        
        if not text_components:
            return []
        
        # Group into lines
        text_components.sort(key=lambda c: c['centroid'][1])
        lines = []
        current_line = [text_components[0]]
        
        for comp in text_components[1:]:
            line_y_center = np.mean([c['centroid'][1] for c in current_line])
            avg_height = np.mean([c['bbox'][3] for c in current_line])
            
            if abs(comp['centroid'][1] - line_y_center) <= avg_height * 0.45:
                current_line.append(comp)
            else:
                lines.append(current_line)
                current_line = [comp]
        
        if current_line:
            lines.append(current_line)
        
        # Segment into words
        word_boxes = []
        for line in lines:
            line.sort(key=lambda c: c['bbox'][0])
            
            char_widths = [c['bbox'][2] for c in line]
            median_char_width = np.median(char_widths) if char_widths else 10
            word_gap = median_char_width * 0.6
            
            current_word = [line[0]]
            
            for i in range(1, len(line)):
                prev = line[i-1]
                curr = line[i]
                gap = curr['bbox'][0] - (prev['bbox'][0] + prev['bbox'][2])
                
                if gap <= word_gap:
                    current_word.append(curr)
                else:
                    if current_word:
                        word_box = self._merge_component_boxes(current_word)
                        word_box = self._add_padding(word_box, img_w, img_h)
                        word_boxes.append(word_box)
                    current_word = [curr]
            
            if current_word:
                word_box = self._merge_component_boxes(current_word)
                word_box = self._add_padding(word_box, img_w, img_h)
                word_boxes.append(word_box)
        
        return word_boxes
    
    def _merge_component_boxes(self, components):
        """Merge component bounding boxes"""
        if not components:
            return None
        
        x_min = min(c['bbox'][0] for c in components)
        y_min = min(c['bbox'][1] for c in components)
        x_max = max(c['bbox'][0] + c['bbox'][2] for c in components)
        y_max = max(c['bbox'][1] + c['bbox'][3] for c in components)
        
        return (x_min, y_min, x_max - x_min, y_max - y_min)
    
    def _add_padding(self, box, img_w, img_h):
        """Add padding to a box with boundary checks"""
        if box is None:
            return None
        x, y, w, h = box
        padding = self._auto_padding if self._auto_padding is not None else 10
        x_pad = max(0, x - padding)
        y_pad = max(0, y - padding)
        w_pad = min(img_w - x_pad, w + 2 * padding)
        h_pad = min(img_h - y_pad, h + 2 * padding)
        return (x_pad, y_pad, w_pad, h_pad)
    
    def _merge_overlapping_boxes(self, boxes, median_height):
        """Merge boxes that overlap vertically"""
        if not boxes:
            return []
        
        boxes = sorted(boxes, key=lambda b: b[1])
        merged = []
        current_box = boxes[0]
        
        for next_box in boxes[1:]:
            curr_y1, curr_y2 = current_box[1], current_box[1] + current_box[3]
            next_y1, next_y2 = next_box[1], next_box[1] + next_box[3]
            
            overlap_start = max(curr_y1, next_y1)
            overlap_end = min(curr_y2, next_y2)
            overlap = max(0, overlap_end - overlap_start)
            
            curr_height = current_box[3]
            next_height = next_box[3]
            min_height = min(curr_height, next_height)
            
            if overlap > min_height * 0.4:
                x_min = min(current_box[0], next_box[0])
                y_min = min(current_box[1], next_box[1])
                x_max = max(current_box[0] + current_box[2], next_box[0] + next_box[2])
                y_max = max(current_box[1] + current_box[3], next_box[1] + next_box[3])
                current_box = (x_min, y_min, x_max - x_min, y_max - y_min)
            else:
                merged.append(current_box)
                current_box = next_box
        
        merged.append(current_box)
        return merged
    
    def is_multiline(self, image_path, threshold=2):
        """Check if image contains multiple lines"""
        boxes = self.detect_lines(image_path)
        return len(boxes) >= threshold