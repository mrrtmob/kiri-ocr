"""
Legacy Text Detector using Classic Computer Vision.

This module provides text detection using traditional image processing
techniques for environments without deep learning dependencies.
"""
import cv2
import numpy as np
import os
from typing import List, Tuple, Optional, Dict, Union
from pathlib import Path
import warnings

from ..base import TextBox, DetectionLevel


class ImageProcessingTextDetector:
    """
    Advanced Text Detector (Classic Computer Vision Approach).
    
    Detects text regions robustly across ALL background/text color conditions
    using multiple detection strategies:
    
    1. Multi-channel binarization (RGB + HSV + LAB)
    2. MSER (Maximally Stable Extremal Regions)
    3. Gradient/Edge-based detection
    4. Stroke Width Transform approximation
    5. Connected Component Analysis with smart filtering
    6. Projection profile analysis for line segmentation
    
    Example:
        detector = ImageProcessingTextDetector()
        lines = detector.detect_lines("image.png")
        words = detector.detect_words("image.png")
        hierarchy = detector.detect_all("image.png")  # Full hierarchy
    """
    
    def __init__(
        self,
        padding: Optional[int] = None,
        min_text_height: int = 6,
        max_text_height: Optional[int] = None,
        min_text_width: int = 2,
        min_confidence: float = 0.3,
        use_mser: bool = True,
        use_gradient: bool = True,
        use_color_channels: bool = True,
        scales: Tuple[float, ...] = (1.0,),
        debug: bool = False
    ):
        """
        Initialize the ImageProcessingTextDetector.
        
        Args:
            padding: Pixels to add around detected boxes (None = auto-calculate)
            min_text_height: Minimum text height in pixels
            max_text_height: Maximum text height (None = 50% of image height)
            min_text_width: Minimum text width in pixels
            min_confidence: Minimum confidence threshold for detections
            use_mser: Enable MSER-based detection
            use_gradient: Enable gradient-based detection  
            use_color_channels: Enable multi-color channel analysis
            scales: Image scales to process (for multi-scale detection)
            debug: Enable debug output and visualization
        """
        self.padding = padding
        self.min_text_height = min_text_height
        self.max_text_height = max_text_height
        self.min_text_width = min_text_width
        self.min_confidence = min_confidence
        self.use_mser = use_mser
        self.use_gradient = use_gradient
        self.use_color_channels = use_color_channels
        self.scales = scales
        self.debug = debug
        
        # Runtime state
        self._auto_padding = None
        self._median_char_height = None
        self._median_char_width = None
        self._debug_images = {}
    
    # ==================== PUBLIC API ====================
    
    def detect_lines(self, image: Union[str, Path, np.ndarray]) -> List[Tuple[int, int, int, int]]:
        """
        Detect text lines in the image.
        
        Args:
            image: Image path or numpy array (BGR format)
            
        Returns:
            List of bounding boxes [(x, y, w, h), ...] sorted top to bottom
        """
        boxes = self._detect(image, level=DetectionLevel.LINE)
        return [box.bbox for box in boxes]
    
    def detect_words(self, image: Union[str, Path, np.ndarray]) -> List[Tuple[int, int, int, int]]:
        """
        Detect words in the image.
        
        Args:
            image: Image path or numpy array (BGR format)
            
        Returns:
            List of bounding boxes [(x, y, w, h), ...] sorted by reading order
        """
        boxes = self._detect(image, level=DetectionLevel.WORD)
        return [box.bbox for box in boxes]
    
    def detect_characters(self, image: Union[str, Path, np.ndarray]) -> List[Tuple[int, int, int, int]]:
        """
        Detect individual characters in the image.
        
        Args:
            image: Image path or numpy array (BGR format)
            
        Returns:
            List of bounding boxes [(x, y, w, h), ...]
        """
        boxes = self._detect(image, level=DetectionLevel.CHARACTER)
        return [box.bbox for box in boxes]
    
    def detect_blocks(self, image: Union[str, Path, np.ndarray]) -> List[Tuple[int, int, int, int]]:
        """
        Detect text blocks/paragraphs in the image.
        
        Args:
            image: Image path or numpy array (BGR format)
            
        Returns:
            List of bounding boxes [(x, y, w, h), ...]
        """
        boxes = self._detect(image, level=DetectionLevel.BLOCK)
        return [box.bbox for box in boxes]
    
    def detect_all(self, image: Union[str, Path, np.ndarray]) -> List[TextBox]:
        """
        Detect full text hierarchy (blocks -> lines -> words).
        
        Args:
            image: Image path or numpy array
            
        Returns:
            List of TextBox objects with nested children
        """
        return self._detect(image, level=DetectionLevel.BLOCK, build_hierarchy=True)
    
    def is_multiline(self, image: Union[str, Path, np.ndarray], threshold: int = 2) -> bool:
        """Check if image contains multiple text lines."""
        lines = self.detect_lines(image)
        return len(lines) >= threshold
    
    def get_debug_images(self) -> Dict[str, np.ndarray]:
        """Get debug visualization images (requires debug=True)."""
        return self._debug_images.copy()
    
    # ==================== MAIN DETECTION PIPELINE ====================
    
    def _detect(
        self, 
        image: Union[str, Path, np.ndarray],
        level: DetectionLevel = DetectionLevel.LINE,
        build_hierarchy: bool = False
    ) -> List[TextBox]:
        """Main detection pipeline."""
        
        # Load image
        img = self._load_image(image)
        if img is None:
            return []
        
        img_h, img_w = img.shape[:2]
        
        # Set max text height if not specified
        max_h = self.max_text_height or int(img_h * 0.5)
        
        # Reset debug images
        self._debug_images = {}
        
        # Step 1: Get all candidate text components using multiple methods
        all_components = []
        
        for scale in self.scales:
            if scale != 1.0:
                scaled_img = cv2.resize(img, None, fx=scale, fy=scale)
            else:
                scaled_img = img
            
            # Method 1: Multi-channel binarization + connected components
            cc_components = self._detect_by_connected_components(scaled_img)
            all_components.extend(self._rescale_components(cc_components, 1.0/scale))
            
            # Method 2: MSER detection
            if self.use_mser:
                mser_components = self._detect_by_mser(scaled_img)
                all_components.extend(self._rescale_components(mser_components, 1.0/scale))
            
            # Method 3: Gradient-based detection
            if self.use_gradient:
                grad_components = self._detect_by_gradient(scaled_img)
                all_components.extend(self._rescale_components(grad_components, 1.0/scale))
        
        if not all_components:
            return []
        
        # Step 2: Filter and deduplicate components
        filtered_components = self._filter_components(all_components, img_w, img_h, max_h)
        
        if not filtered_components:
            return []
        
        # Step 3: Estimate text metrics
        self._estimate_text_metrics(filtered_components)
        
        # Step 4: Group into lines
        lines = self._group_into_lines(filtered_components)
        
        # Step 5: Build output based on requested level
        if level == DetectionLevel.CHARACTER:
            boxes = [self._component_to_textbox(c, DetectionLevel.CHARACTER) for c in filtered_components]
            boxes = sorted(boxes, key=lambda b: (b.y, b.x))
        
        elif level == DetectionLevel.WORD:
            boxes = self._segment_lines_to_words(lines, img_w, img_h)
        
        elif level == DetectionLevel.LINE:
            boxes = self._create_line_boxes(lines, img_w, img_h)
        
        elif level == DetectionLevel.BLOCK:
            line_boxes = self._create_line_boxes(lines, img_w, img_h)
            boxes = self._group_lines_into_blocks(line_boxes, img_w, img_h)
            
            if build_hierarchy:
                # Build full hierarchy
                for block in boxes:
                    block_lines = [lb for lb in line_boxes if self._box_contains(block, lb)]
                    for line in block_lines:
                        line_words = self._segment_single_line_to_words(
                            [c for c in filtered_components 
                             if self._point_in_box((c['cx'], c['cy']), line.bbox)],
                            img_w, img_h
                        )
                        line.children = line_words
                    block.children = block_lines
        
        else:
            boxes = self._create_line_boxes(lines, img_w, img_h)
        
        # Filter by confidence
        boxes = [b for b in boxes if b.confidence >= self.min_confidence]
        
        return boxes
    
    # ==================== DETECTION METHODS ====================
    
    def _detect_by_connected_components(self, img: np.ndarray) -> List[Dict]:
        """Detect text components using connected component analysis on multiple binarizations."""
        
        components = []
        binary_candidates = self._get_binary_candidates(img)
        
        # Score and select best candidates
        scored_binaries = []
        for name, binary in binary_candidates:
            score, stats = self._score_binarization(binary, img.shape[:2])
            scored_binaries.append((score, name, binary, stats))
        
        # Sort by score and use top candidates
        scored_binaries.sort(reverse=True, key=lambda x: x[0])
        
        # Use top 3 binarizations
        for score, name, binary, stats in scored_binaries[:3]:
            if score > 0:
                comps = self._extract_components(binary, name, score)
                components.extend(comps)
                
                if self.debug:
                    self._debug_images[f'binary_{name}'] = binary
        
        return components
    
    def _detect_by_mser(self, img: np.ndarray) -> List[Dict]:
        """Detect text regions using MSER (Maximally Stable Extremal Regions)."""
        
        components = []
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        
        # Create MSER detector with text-optimized parameters
        mser = cv2.MSER_create(
            delta=5,
            min_area=30,
            max_area=14400,
            max_variation=0.25,
            min_diversity=0.2,
            max_evolution=200,
            area_threshold=1.01,
            min_margin=0.003,
            edge_blur_size=5
        )
        
        # Detect on both original and inverted
        for invert in [False, True]:
            img_proc = 255 - gray if invert else gray
            
            try:
                regions, _ = mser.detectRegions(img_proc)
                
                for region in regions:
                    x, y, w, h = cv2.boundingRect(region)
                    
                    if w >= self.min_text_width and h >= self.min_text_height:
                        area = cv2.contourArea(region.reshape(-1, 1, 2))
                        hull = cv2.convexHull(region.reshape(-1, 1, 2))
                        hull_area = cv2.contourArea(hull)
                        
                        # Solidity check (text usually has medium solidity)
                        solidity = area / hull_area if hull_area > 0 else 0
                        
                        if 0.2 < solidity < 0.95:
                            components.append({
                                'bbox': (x, y, w, h),
                                'cx': x + w/2,
                                'cy': y + h/2,
                                'area': area,
                                'source': 'mser',
                                'confidence': 0.7 * solidity
                            })
            except cv2.error:
                pass
        
        return components
    
    def _detect_by_gradient(self, img: np.ndarray) -> List[Dict]:
        """Detect text using gradient/edge-based approach (like Stroke Width Transform)."""
        
        components = []
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        
        # Compute gradients
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Gradient magnitude
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        magnitude = (magnitude / magnitude.max() * 255).astype(np.uint8)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Dilate edges to connect nearby strokes
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
        dilated = cv2.dilate(edges, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            if w >= self.min_text_width and h >= self.min_text_height:
                # Check if region has consistent stroke width (text characteristic)
                roi_mag = magnitude[y:y+h, x:x+w]
                
                if roi_mag.size > 0:
                    # Text typically has bimodal gradient distribution
                    non_zero = roi_mag[roi_mag > 20]
                    if len(non_zero) > 10:
                        stroke_consistency = 1.0 - (np.std(non_zero) / (np.mean(non_zero) + 1e-6))
                        stroke_consistency = max(0, min(1, stroke_consistency))
                        
                        aspect_ratio = w / h
                        # Text chars usually have aspect ratio 0.1 - 10
                        if 0.05 < aspect_ratio < 15:
                            components.append({
                                'bbox': (x, y, w, h),
                                'cx': x + w/2,
                                'cy': y + h/2,
                                'area': w * h,
                                'source': 'gradient',
                                'confidence': 0.5 * stroke_consistency
                            })
        
        if self.debug:
            self._debug_images['gradient_edges'] = dilated
        
        return components
    
    # ==================== BINARIZATION ====================
    
    def _get_binary_candidates(self, img: np.ndarray) -> List[Tuple[str, np.ndarray]]:
        """Generate multiple binarization candidates from different color spaces and methods."""
        
        candidates = []
        
        # Convert to different color spaces
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        
        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # === Grayscale binarizations ===
        
        # Otsu's method
        _, otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        candidates.append(('otsu', otsu))
        candidates.append(('otsu_inv', 255 - otsu))
        
        # Adaptive threshold - Gaussian
        adapt_gauss = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 21, 10
        )
        candidates.append(('adaptive_gauss', adapt_gauss))
        candidates.append(('adaptive_gauss_inv', 255 - adapt_gauss))
        
        # Adaptive threshold - Mean (different characteristics)
        adapt_mean = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY, 15, 8
        )
        candidates.append(('adaptive_mean', adapt_mean))
        candidates.append(('adaptive_mean_inv', 255 - adapt_mean))
        
        # Sauvola-like (larger block size)
        sauvola = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 51, 20
        )
        candidates.append(('sauvola', sauvola))
        candidates.append(('sauvola_inv', 255 - sauvola))
        
        # Niblack-like (smaller block, different offset)
        niblack = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY, 11, 5
        )
        candidates.append(('niblack', niblack))
        candidates.append(('niblack_inv', 255 - niblack))
        
        # === Multi-channel binarizations (for colored backgrounds) ===
        
        if self.use_color_channels and len(img.shape) == 3:
            # Process each RGB channel
            for i, channel_name in enumerate(['blue', 'green', 'red']):
                channel = img[:, :, i]
                channel_enhanced = clahe.apply(channel)
                
                _, ch_otsu = cv2.threshold(channel_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                candidates.append((f'{channel_name}_otsu', ch_otsu))
                candidates.append((f'{channel_name}_otsu_inv', 255 - ch_otsu))
            
            # HSV color space (good for colored text/backgrounds)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            
            # Value channel
            v_channel = hsv[:, :, 2]
            v_enhanced = clahe.apply(v_channel)
            _, v_otsu = cv2.threshold(v_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            candidates.append(('hsv_v_otsu', v_otsu))
            candidates.append(('hsv_v_otsu_inv', 255 - v_otsu))
            
            # Saturation channel (helps with colored text on white/gray bg)
            s_channel = hsv[:, :, 1]
            _, s_thresh = cv2.threshold(s_channel, 50, 255, cv2.THRESH_BINARY)
            candidates.append(('hsv_s', s_thresh))
            
            # LAB color space (perceptually uniform)
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l_channel = lab[:, :, 0]
            l_enhanced = clahe.apply(l_channel)
            _, l_otsu = cv2.threshold(l_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            candidates.append(('lab_l_otsu', l_otsu))
            candidates.append(('lab_l_otsu_inv', 255 - l_otsu))
            
            # A and B channels (for colored text)
            for i, ch_name in enumerate(['a', 'b']):
                ch = lab[:, :, i + 1]
                # A and B are centered at 128, threshold at extremes
                _, ch_high = cv2.threshold(ch, 160, 255, cv2.THRESH_BINARY)
                _, ch_low = cv2.threshold(ch, 96, 255, cv2.THRESH_BINARY_INV)
                candidates.append((f'lab_{ch_name}_high', ch_high))
                candidates.append((f'lab_{ch_name}_low', ch_low))
        
        # === Edge-based binarization ===
        
        # Morphological gradient (highlights edges)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        morph_grad = cv2.morphologyEx(enhanced, cv2.MORPH_GRADIENT, kernel)
        _, morph_bin = cv2.threshold(morph_grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        candidates.append(('morph_gradient', morph_bin))
        
        return candidates
    
    def _score_binarization(self, binary: np.ndarray, img_shape: Tuple[int, int]) -> Tuple[float, Dict]:
        """Score a binarization based on how text-like the components are."""
        
        img_h, img_w = img_shape[:2]
        
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        
        if num_labels <= 1:
            return 0, {}
        
        # Analyze components
        heights = []
        widths = []
        areas = []
        aspect_ratios = []
        fill_ratios = []
        valid_count = 0
        
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            
            # Basic size filter
            if w >= 2 and h >= 3 and area >= 6:
                # Skip components touching image border (likely noise/background)
                if x > 0 and y > 0 and x + w < img_w and y + h < img_h:
                    heights.append(h)
                    widths.append(w)
                    areas.append(area)
                    
                    ar = w / h if h > 0 else 0
                    aspect_ratios.append(ar)
                    
                    fill = area / (w * h) if w * h > 0 else 0
                    fill_ratios.append(fill)
                    
                    # Count text-like components
                    if 0.1 < ar < 10 and 0.1 < fill < 0.95 and h < img_h * 0.3:
                        valid_count += 1
        
        if len(heights) < 3:
            return 0, {}
        
        # Calculate metrics
        median_h = np.median(heights)
        median_w = np.median(widths)
        std_h = np.std(heights)
        
        # Score components:
        # 1. Number of text-like components
        count_score = min(valid_count / 10, 1.0)  # Normalize
        
        # 2. Height consistency (text should have similar heights)
        consistency_score = 1.0 / (1.0 + std_h / max(median_h, 1))
        
        # 3. Reasonable median height
        size_score = 1.0 if 8 <= median_h <= 100 else 0.5
        
        # 4. Good aspect ratio distribution
        median_ar = np.median(aspect_ratios)
        ar_score = 1.0 if 0.3 < median_ar < 3 else 0.5
        
        # Combined score
        score = valid_count * consistency_score * size_score * ar_score
        
        stats_dict = {
            'num_components': num_labels - 1,
            'valid_components': valid_count,
            'median_height': median_h,
            'median_width': median_w,
            'height_std': std_h,
            'median_aspect_ratio': median_ar
        }
        
        return score, stats_dict
    
    def _extract_components(self, binary: np.ndarray, source: str, confidence_base: float) -> List[Dict]:
        """Extract connected components from a binary image."""
        
        components = []
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            cx, cy = centroids[i]
            
            if w >= self.min_text_width and h >= self.min_text_height:
                # Calculate component confidence
                aspect_ratio = w / h
                fill_ratio = area / (w * h) if w * h > 0 else 0
                
                # Text-like characteristics
                ar_confidence = 1.0 if 0.15 < aspect_ratio < 8 else 0.5
                fill_confidence = 1.0 if 0.15 < fill_ratio < 0.9 else 0.5
                
                confidence = confidence_base * ar_confidence * fill_confidence * 0.01  # Normalize
                confidence = min(1.0, confidence)
                
                components.append({
                    'bbox': (int(x), int(y), int(w), int(h)),
                    'cx': float(cx),
                    'cy': float(cy),
                    'area': int(area),
                    'source': source,
                    'confidence': confidence
                })
        
        return components
    
    # ==================== COMPONENT PROCESSING ====================
    
    def _rescale_components(self, components: List[Dict], scale: float) -> List[Dict]:
        """Rescale component coordinates."""
        if scale == 1.0:
            return components
        
        rescaled = []
        for comp in components:
            x, y, w, h = comp['bbox']
            rescaled.append({
                'bbox': (int(x * scale), int(y * scale), int(w * scale), int(h * scale)),
                'cx': comp['cx'] * scale,
                'cy': comp['cy'] * scale,
                'area': int(comp['area'] * scale * scale),
                'source': comp['source'],
                'confidence': comp['confidence']
            })
        return rescaled
    
    def _filter_components(
        self, 
        components: List[Dict], 
        img_w: int, 
        img_h: int,
        max_h: int
    ) -> List[Dict]:
        """Filter and deduplicate components."""
        
        if not components:
            return []
        
        # First pass: basic filtering
        filtered = []
        for comp in components:
            x, y, w, h = comp['bbox']
            
            # Size constraints
            if w < self.min_text_width or h < self.min_text_height:
                continue
            if h > max_h or w > img_w * 0.98:
                continue
            
            # Aspect ratio filter (allow wide range for different scripts)
            aspect = w / h if h > 0 else 0
            if aspect < 0.02 or aspect > 50:
                continue
            
            # Not too close to image edges with suspicious sizes
            if (x == 0 or y == 0 or x + w >= img_w or y + h >= img_h):
                if w > img_w * 0.5 or h > img_h * 0.5:
                    continue
            
            filtered.append(comp)
        
        if not filtered:
            return []
        
        # Estimate typical character size for smarter filtering
        heights = [c['bbox'][3] for c in filtered]
        median_h = np.median(heights)
        
        # Second pass: filter by relative size
        size_filtered = []
        for comp in filtered:
            h = comp['bbox'][3]
            # Keep components within reasonable range of median
            if median_h * 0.15 <= h <= median_h * 5:
                size_filtered.append(comp)
        
        # Deduplicate overlapping components (keep highest confidence)
        deduplicated = self._deduplicate_components(size_filtered)
        
        return deduplicated
    
    def _deduplicate_components(self, components: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
        """Remove duplicate/overlapping components using NMS-like approach."""
        
        if len(components) <= 1:
            return components
        
        # Sort by confidence (descending)
        sorted_components = sorted(components, key=lambda c: c['confidence'], reverse=True)
        
        keep = []
        used = set()
        
        for i, comp in enumerate(sorted_components):
            if i in used:
                continue
            
            keep.append(comp)
            
            # Mark overlapping components as used
            for j in range(i + 1, len(sorted_components)):
                if j in used:
                    continue
                
                iou = self._calculate_iou(comp['bbox'], sorted_components[j]['bbox'])
                if iou > iou_threshold:
                    used.add(j)
        
        return keep
    
    def _calculate_iou(self, bbox1: Tuple, bbox2: Tuple) -> float:
        """Calculate Intersection over Union of two bboxes."""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Calculate intersection
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        
        intersection = (xi2 - xi1) * (yi2 - yi1)
        
        # Calculate union
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _estimate_text_metrics(self, components: List[Dict]) -> None:
        """Estimate typical text character size."""
        if not components:
            return
        
        heights = [c['bbox'][3] for c in components]
        widths = [c['bbox'][2] for c in components]
        
        self._median_char_height = np.median(heights)
        self._median_char_width = np.median(widths)
        
        # Set auto padding
        if self.padding is None:
            self._auto_padding = max(2, int(self._median_char_height * 0.15))
        else:
            self._auto_padding = self.padding
    
    # ==================== LINE GROUPING ====================
    
    def _group_into_lines(self, components: List[Dict]) -> List[List[Dict]]:
        """Group components into text lines using baseline clustering."""
        
        if not components:
            return []
        
        # Sort by vertical position
        sorted_comps = sorted(components, key=lambda c: c['cy'])
        
        # Estimate line height for clustering threshold
        if self._median_char_height:
            line_threshold = self._median_char_height * 0.6
        else:
            heights = [c['bbox'][3] for c in components]
            line_threshold = np.median(heights) * 0.6
        
        # Cluster into lines
        lines = []
        current_line = [sorted_comps[0]]
        
        for comp in sorted_comps[1:]:
            # Calculate line's current baseline (use weighted average y-center)
            line_y_positions = [c['cy'] for c in current_line]
            line_y = np.mean(line_y_positions)
            
            # Use adaptive threshold based on current line's height
            line_heights = [c['bbox'][3] for c in current_line]
            adaptive_threshold = max(line_threshold, np.mean(line_heights) * 0.5)
            
            if abs(comp['cy'] - line_y) <= adaptive_threshold:
                current_line.append(comp)
            else:
                lines.append(current_line)
                current_line = [comp]
        
        if current_line:
            lines.append(current_line)
        
        # Sort components within each line by x-coordinate
        for line in lines:
            line.sort(key=lambda c: c['bbox'][0])
        
        return lines
    
    # ==================== OUTPUT GENERATION ====================
    
    def _create_line_boxes(self, lines: List[List[Dict]], img_w: int, img_h: int) -> List[TextBox]:
        """Create line bounding boxes from component groups."""
        
        boxes = []
        for line in lines:
            if not line:
                continue
            
            # Calculate bounding box
            x_min = min(c['bbox'][0] for c in line)
            y_min = min(c['bbox'][1] for c in line)
            x_max = max(c['bbox'][0] + c['bbox'][2] for c in line)
            y_max = max(c['bbox'][1] + c['bbox'][3] for c in line)
            
            # Average confidence
            avg_confidence = np.mean([c['confidence'] for c in line])
            
            # Add padding
            padding = self._auto_padding or 2
            x = max(0, x_min - padding)
            y = max(0, y_min - padding)
            w = min(img_w - x, (x_max - x_min) + 2 * padding)
            h = min(img_h - y, (y_max - y_min) + 2 * padding)
            
            boxes.append(TextBox(
                x=x, y=y, width=w, height=h,
                confidence=avg_confidence,
                level=DetectionLevel.LINE
            ))
        
        # Merge overlapping line boxes
        boxes = self._merge_overlapping_boxes(boxes)
        
        # Sort by y-coordinate
        boxes.sort(key=lambda b: b.y)
        
        return boxes
    
    def _segment_lines_to_words(self, lines: List[List[Dict]], img_w: int, img_h: int) -> List[TextBox]:
        """Segment lines into words based on spacing."""
        
        all_words = []
        
        for line in lines:
            words = self._segment_single_line_to_words(line, img_w, img_h)
            all_words.extend(words)
        
        # Sort by reading order (top to bottom, left to right)
        all_words.sort(key=lambda b: (b.y, b.x))
        
        return all_words
    
    def _segment_single_line_to_words(self, line: List[Dict], img_w: int, img_h: int) -> List[TextBox]:
        """Segment a single line into words."""
        
        if not line:
            return []
        
        # Sort by x-coordinate
        line = sorted(line, key=lambda c: c['bbox'][0])
        
        # Calculate inter-character gaps
        gaps = []
        for i in range(1, len(line)):
            prev = line[i-1]
            curr = line[i]
            gap = curr['bbox'][0] - (prev['bbox'][0] + prev['bbox'][2])
            gaps.append(gap)
        
        if not gaps:
            # Single character = single word
            return [self._components_to_word_box(line, img_w, img_h)]
        
        # Determine word gap threshold
        # Use median width as reference
        char_widths = [c['bbox'][2] for c in line]
        median_width = np.median(char_widths)
        
        # Word gap is typically larger than inter-character gap
        # Use adaptive threshold
        if len(gaps) >= 3:
            gap_median = np.median(gaps)
            gap_std = np.std(gaps)
            word_gap_threshold = gap_median + gap_std
        else:
            word_gap_threshold = median_width * 0.5
        
        # Ensure reasonable threshold
        word_gap_threshold = max(word_gap_threshold, median_width * 0.3)
        word_gap_threshold = min(word_gap_threshold, median_width * 2.0)
        
        # Segment into words
        words = []
        current_word = [line[0]]
        
        for i in range(1, len(line)):
            gap = line[i]['bbox'][0] - (line[i-1]['bbox'][0] + line[i-1]['bbox'][2])
            
            if gap <= word_gap_threshold:
                current_word.append(line[i])
            else:
                if current_word:
                    words.append(self._components_to_word_box(current_word, img_w, img_h))
                current_word = [line[i]]
        
        if current_word:
            words.append(self._components_to_word_box(current_word, img_w, img_h))
        
        return words
    
    def _components_to_word_box(self, components: List[Dict], img_w: int, img_h: int) -> TextBox:
        """Create a word TextBox from components."""
        
        x_min = min(c['bbox'][0] for c in components)
        y_min = min(c['bbox'][1] for c in components)
        x_max = max(c['bbox'][0] + c['bbox'][2] for c in components)
        y_max = max(c['bbox'][1] + c['bbox'][3] for c in components)
        
        avg_confidence = np.mean([c['confidence'] for c in components])
        
        # Add padding
        padding = self._auto_padding or 2
        x = max(0, x_min - padding)
        y = max(0, y_min - padding)
        w = min(img_w - x, (x_max - x_min) + 2 * padding)
        h = min(img_h - y, (y_max - y_min) + 2 * padding)
        
        return TextBox(
            x=x, y=y, width=w, height=h,
            confidence=avg_confidence,
            level=DetectionLevel.WORD
        )
    
    def _group_lines_into_blocks(self, line_boxes: List[TextBox], img_w: int, img_h: int) -> List[TextBox]:
        """Group lines into text blocks/paragraphs."""
        
        if not line_boxes:
            return []
        
        if len(line_boxes) == 1:
            box = line_boxes[0]
            return [TextBox(
                x=box.x, y=box.y, width=box.width, height=box.height,
                confidence=box.confidence,
                level=DetectionLevel.BLOCK
            )]
        
        # Sort by y-coordinate
        sorted_lines = sorted(line_boxes, key=lambda b: b.y)
        
        # Calculate typical line spacing
        line_gaps = []
        for i in range(1, len(sorted_lines)):
            gap = sorted_lines[i].y - (sorted_lines[i-1].y + sorted_lines[i-1].height)
            line_gaps.append(gap)
        
        if line_gaps:
            median_gap = np.median(line_gaps)
            # Block gap threshold = large line spacing
            block_gap_threshold = max(median_gap * 2, self._median_char_height or 20)
        else:
            block_gap_threshold = self._median_char_height * 2 if self._median_char_height else 40
        
        # Group into blocks
        blocks = []
        current_block = [sorted_lines[0]]
        
        for i in range(1, len(sorted_lines)):
            prev_line = current_block[-1]
            curr_line = sorted_lines[i]
            
            gap = curr_line.y - (prev_line.y + prev_line.height)
            
            # Check horizontal alignment too
            x_overlap = self._calculate_x_overlap(prev_line, curr_line)
            
            if gap <= block_gap_threshold and x_overlap > 0.3:
                current_block.append(curr_line)
            else:
                blocks.append(self._lines_to_block_box(current_block))
                current_block = [curr_line]
        
        if current_block:
            blocks.append(self._lines_to_block_box(current_block))
        
        return blocks
    
    def _lines_to_block_box(self, lines: List[TextBox]) -> TextBox:
        """Create a block TextBox from lines."""
        
        x_min = min(l.x for l in lines)
        y_min = min(l.y for l in lines)
        x_max = max(l.x + l.width for l in lines)
        y_max = max(l.y + l.height for l in lines)
        
        avg_confidence = np.mean([l.confidence for l in lines])
        
        return TextBox(
            x=x_min, y=y_min,
            width=x_max - x_min,
            height=y_max - y_min,
            confidence=avg_confidence,
            level=DetectionLevel.BLOCK,
            children=lines
        )
    
    def _calculate_x_overlap(self, box1: TextBox, box2: TextBox) -> float:
        """Calculate horizontal overlap ratio between two boxes."""
        
        x1_start, x1_end = box1.x, box1.x + box1.width
        x2_start, x2_end = box2.x, box2.x + box2.width
        
        overlap_start = max(x1_start, x2_start)
        overlap_end = min(x1_end, x2_end)
        
        if overlap_end <= overlap_start:
            return 0.0
        
        overlap = overlap_end - overlap_start
        min_width = min(box1.width, box2.width)
        
        return overlap / min_width if min_width > 0 else 0.0
    
    def _merge_overlapping_boxes(self, boxes: List[TextBox]) -> List[TextBox]:
        """Merge vertically overlapping boxes."""
        
        if len(boxes) <= 1:
            return boxes
        
        # Sort by y-coordinate
        boxes = sorted(boxes, key=lambda b: b.y)
        
        merged = []
        current = boxes[0]
        
        for next_box in boxes[1:]:
            # Check vertical overlap
            curr_y1, curr_y2 = current.y, current.y + current.height
            next_y1, next_y2 = next_box.y, next_box.y + next_box.height
            
            overlap_start = max(curr_y1, next_y1)
            overlap_end = min(curr_y2, next_y2)
            overlap = max(0, overlap_end - overlap_start)
            
            min_height = min(current.height, next_box.height)
            
            if overlap > min_height * 0.3:
                # Merge boxes
                x_min = min(current.x, next_box.x)
                y_min = min(current.y, next_box.y)
                x_max = max(current.x + current.width, next_box.x + next_box.width)
                y_max = max(current.y + current.height, next_box.y + next_box.height)
                
                avg_conf = (current.confidence + next_box.confidence) / 2
                
                current = TextBox(
                    x=x_min, y=y_min,
                    width=x_max - x_min,
                    height=y_max - y_min,
                    confidence=avg_conf,
                    level=current.level
                )
            else:
                merged.append(current)
                current = next_box
        
        merged.append(current)
        return merged
    
    # ==================== UTILITIES ====================
    
    def _load_image(self, image: Union[str, Path, np.ndarray]) -> Optional[np.ndarray]:
        """Load image from path or return numpy array."""
        
        if isinstance(image, np.ndarray):
            return image
        
        img = cv2.imread(str(image))
        if img is None:
            warnings.warn(f"Could not load image: {image}")
        return img
    
    def _component_to_textbox(self, comp: Dict, level: DetectionLevel) -> TextBox:
        """Convert component dict to TextBox."""
        x, y, w, h = comp['bbox']
        return TextBox(
            x=x, y=y, width=w, height=h,
            confidence=comp['confidence'],
            level=level
        )
    
    def _box_contains(self, outer: TextBox, inner: TextBox) -> bool:
        """Check if outer box contains inner box."""
        return (outer.x <= inner.x and 
                outer.y <= inner.y and
                outer.x + outer.width >= inner.x + inner.width and
                outer.y + outer.height >= inner.y + inner.height)
    
    def _point_in_box(self, point: Tuple[float, float], bbox: Tuple[int, int, int, int]) -> bool:
        """Check if point is inside bbox."""
        px, py = point
        x, y, w, h = bbox
        return x <= px <= x + w and y <= py <= y + h
