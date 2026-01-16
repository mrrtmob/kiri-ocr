import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

class TextDetector:
    """Detect text regions using Tesseract's actual approach - language-agnostic"""
    
    def __init__(self, padding=None):
        """
        Args:
            padding: Pixels to add around detected boxes (default: None for auto)
                    If None, padding will be automatically calculated based on text size
        """
        self.padding = padding
        self._auto_padding = None
    
    def detect_lines(self, image_path):
        """
        Detect text lines using Tesseract's actual Page Layout Analysis:
        1. Otsu binarization
        2. Connected components with stats
        3. Estimate median text height from components
        4. Filter noise based on text height
        5. Find baselines and group into text lines
        6. Merge overlapping regions
        
        Returns:
            List of bounding boxes [(x, y, w, h), ...]
        """
        # Load and preprocess
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        img_h, img_w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Step 1: Otsu binarization (foreground detection)
        # auto-detects if it needs to invert
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Check if we got mostly white (wrong polarity)
        if np.mean(binary) > 127:
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            binary = 255 - binary
        
        # Step 2: Connected components analysis
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        
        # Step 3: Estimate typical text height
        # Collect component heights
        heights = []
        widths = []
        valid_components = []
        
        for i in range(1, num_labels):  # Skip background
            x, y, w, h, area = stats[i]
            
            # Basic sanity filters (remove 1-pixel noise)
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
        
        # Tesseract's approach: use median height as typical character height
        median_height = np.median(heights)
        median_width = np.median(widths)
        
        # Auto-calculate padding based on median text height
        # typically uses ~10-20% of text height as padding
        if self.padding is None:
            self._auto_padding = max(2, int(median_height * 0.15))
        else:
            self._auto_padding = self.padding
        
        # Step 4: Filter components
        # Keep components that are reasonable relative to median text size
        # uses: 0.25 * median < height < 3.0 * median
        min_h = max(3, median_height * 0.25)
        max_h = median_height * 3.0
        min_w = max(2, median_width * 0.1)
        max_w = img_w * 0.95  # Not entire width
        
        text_components = []
        for comp in valid_components:
            x, y, w, h = comp['bbox']
            
            # Tesseract's blob filtering
            if min_h <= h <= max_h and min_w <= w <= max_w:
                # Aspect ratio check (not too extreme)
                aspect = w / h if h > 0 else 0
                if 0.05 < aspect < 20:  # Reasonable character proportions
                    # Additional filter: remove very small isolated components
                    # (likely diacritics that should be grouped with main text)
                    # Skip components that are less than 30% of median height
                    if h >= median_height * 0.3:
                        text_components.append(comp)
        
        if not text_components:
            return []
        
        # Step 5: Line finding using baseline clustering
        # Group components by vertical position (baseline detection)
        text_components.sort(key=lambda c: c['centroid'][1])
        
        lines = []
        current_line = [text_components[0]]
        
        for comp in text_components[1:]:
            # Get current line's vertical extent
            line_y_values = [c['centroid'][1] for c in current_line]
            line_y_center = np.mean(line_y_values)
            line_heights = [c['bbox'][3] for c in current_line]
            avg_line_height = np.mean(line_heights)
            
            comp_y = comp['centroid'][1]
            
            # For Khmer text with diacritics, use slightly larger tolerance
            # Use 0.45 to group base characters with nearby components
            tolerance = avg_line_height * 0.45
            
            if abs(comp_y - line_y_center) <= tolerance:
                current_line.append(comp)
            else:
                lines.append(current_line)
                current_line = [comp]
        
        if current_line:
            lines.append(current_line)
        
        # Step 6: Create bounding boxes for each line
        line_boxes = []
        for line in lines:
            if not line:
                continue
            
            # Get line bounding box
            x_min = min(c['bbox'][0] for c in line)
            y_min = min(c['bbox'][1] for c in line)
            x_max = max(c['bbox'][0] + c['bbox'][2] for c in line)
            y_max = max(c['bbox'][1] + c['bbox'][3] for c in line)
            
            # Add padding (auto-calculated based on text size)
            x_pad = max(0, x_min - self._auto_padding)
            y_pad = max(0, y_min - self._auto_padding)
            w_pad = min(img_w - x_pad, (x_max - x_min) + 2 * self._auto_padding)
            h_pad = min(img_h - y_pad, (y_max - y_min) + 2 * self._auto_padding)
            
            line_boxes.append((x_pad, y_pad, w_pad, h_pad))
        
        # Merge overlapping boxes (fix duplicate detections)
        line_boxes = self._merge_overlapping_boxes(line_boxes, median_height)
        
        # Sort top to bottom
        line_boxes = sorted(line_boxes, key=lambda b: b[1])
        
        return line_boxes
    
    def detect_words(self, image_path):
        """
        Detect words using Tesseract's approach:
        1. Find all text components
        2. Estimate typical character width
        3. Group into lines
        4. Within lines, cluster by horizontal spacing
        
        Returns:
            List of word bounding boxes
        """
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        img_h, img_w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Binarization
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        if np.mean(binary) > 127:
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            binary = 255 - binary
        
        # Connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        
        # Collect and filter components
        heights = []
        widths = []
        valid_components = []
        
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            if w >= 2 and h >= 3 and area >= 6:
                heights.append(h)
                widths.append(w)
                valid_components.append({
                    'bbox': (x, y, w, h),
                    'centroid': centroids[i]
                })
        
        if not valid_components:
            return []
        
        # Estimate text size
        median_height = np.median(heights)
        median_width = np.median(widths)
        
        # Auto-calculate padding based on median text height
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
                if 0.05 < aspect < 20:
                    # Filter out very small components (diacritics)
                    if h >= median_height * 0.3:
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
            
            # Use 0.45 tolerance for Khmer diacritics
            if abs(comp['centroid'][1] - line_y_center) <= avg_height * 0.45:
                current_line.append(comp)
            else:
                lines.append(current_line)
                current_line = [comp]
        
        if current_line:
            lines.append(current_line)
        
        # Segment lines into words
        word_boxes = []
        for line in lines:
            # Sort by X position
            line.sort(key=lambda c: c['bbox'][0])
            
            # word spacing: typically 0.3 to 1.0 * median character width
            # Space between words is larger than between characters
            char_widths = [c['bbox'][2] for c in line]
            median_char_width = np.median(char_widths) if char_widths else 10
            
            # Word gap threshold
            word_gap = median_char_width * 0.6
            
            current_word = [line[0]]
            
            for i in range(1, len(line)):
                prev = line[i-1]
                curr = line[i]
                
                # Gap between characters
                gap = curr['bbox'][0] - (prev['bbox'][0] + prev['bbox'][2])
                
                if gap <= word_gap:
                    # Same word
                    current_word.append(curr)
                else:
                    # New word
                    if current_word:
                        word_box = self._merge_component_boxes(current_word)
                        word_box = self._add_padding(word_box, img_w, img_h)
                        word_boxes.append(word_box)
                    current_word = [curr]
            
            # Last word
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
        """Merge boxes that overlap vertically (same text line detected multiple times)"""
        if not boxes:
            return []
        
        # Sort by Y coordinate
        boxes = sorted(boxes, key=lambda b: b[1])
        merged = []
        current_box = boxes[0]
        
        for next_box in boxes[1:]:
            # Check vertical overlap
            curr_y1, curr_y2 = current_box[1], current_box[1] + current_box[3]
            next_y1, next_y2 = next_box[1], next_box[1] + next_box[3]
            
            # Calculate overlap
            overlap_start = max(curr_y1, next_y1)
            overlap_end = min(curr_y2, next_y2)
            overlap = max(0, overlap_end - overlap_start)
            
            # Calculate heights
            curr_height = current_box[3]
            next_height = next_box[3]
            
            # More strict merging: only merge if significant overlap
            # Merge if overlap is more than 40% of the smaller box height
            min_height = min(curr_height, next_height)
            if overlap > min_height * 0.4:
                # Merge: take union of both boxes
                x_min = min(current_box[0], next_box[0])
                y_min = min(current_box[1], next_box[1])
                x_max = max(current_box[0] + current_box[2], next_box[0] + next_box[2])
                y_max = max(current_box[1] + current_box[3], next_box[1] + next_box[3])
                current_box = (x_min, y_min, x_max - x_min, y_max - y_min)
            else:
                # No significant overlap, save current and move to next
                merged.append(current_box)
                current_box = next_box
        
        # Add the last box
        merged.append(current_box)
        return merged
    
    def is_multiline(self, image_path, threshold=2):
        """Check if image contains multiple lines"""
        boxes = self.detect_lines(image_path)
        return len(boxes) >= threshold