"""
Realistic Document Dataset Generator for Text Detection (CRAFT-style).

This module provides tools to generate synthetic training data for CRAFT-based
text detection models with realistic multi-line document layouts.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import os
import json
import random
from typing import Union, Dict, List, Tuple
from pathlib import Path


class MultilingualDatasetGenerator:
    """Dataset generator for realistic document-style text with multiple lines."""
    
    def __init__(self, output_dir='dataset', image_width=512, image_height=512):
        self.output_dir = output_dir
        self.image_width = image_width
        self.image_height = image_height
        
        # Create output directories
        os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'annotations'), exist_ok=True)
        
    def test_font_support(self, font, text_sample: str) -> bool:
        """Test if font actually renders text (not boxes)."""
        try:
            # Create small test image
            test_img = Image.new('RGB', (100, 50), color=(255, 255, 255))
            test_draw = ImageDraw.Draw(test_img)
            
            # Try to render text
            test_draw.text((10, 10), text_sample[:5], font=font, fill=(0, 0, 0))
            
            # Check if we got actual pixels (not just boxes)
            pixels = np.array(test_img)
            # If there's variation in pixel values, font is rendering
            return pixels.std() > 1.0
        except:
            return False

    def is_text_supported(self, font, text):
        """Check if font supports the characters in text."""
        try:
            notdef_char = '\uFFFF'
            try:
                notdef_mask = font.getmask(notdef_char)
                notdef_bbox = font.getbbox(notdef_char)
            except:
                return True

            for char in text:
                if char.isspace(): 
                    continue
                
                try:
                    char_bbox = font.getbbox(char)
                    if char_bbox == notdef_bbox:
                        char_mask = font.getmask(char)
                        if bytes(char_mask) == bytes(notdef_mask):
                            return False
                except:
                    continue
            return True
        except:
            return True

    def load_text_lines(self, text_file: Union[str, List[str]]) -> List[str]:
        """Load text lines from file(s)."""
        if isinstance(text_file, str):
            text_file = [text_file]
        
        lines = []
        for file in text_file:
            if not os.path.exists(file):
                print(f"Warning: Text file '{file}' not found, skipping...")
                continue
            with open(file, 'r', encoding='utf-8') as f:
                file_lines = [line.strip() for line in f.readlines() if line.strip()]
                lines.extend(file_lines)
                print(f"  Loaded {len(file_lines)} lines from {file}")
        return lines
    
    def get_font_list(self, font_dir: str) -> List[str]:
        """Get list of font files."""
        font_files = []
        if not os.path.isdir(font_dir):
            print(f"Warning: Font directory '{font_dir}' not found")
            return font_files
            
        for file in os.listdir(font_dir):
            if file.endswith(('.ttf', '.otf', '.TTF', '.OTF')):
                font_files.append(os.path.join(font_dir, file))
        return font_files
    
    def create_sentence(self, text_pool: List[str], max_words: int = None) -> str:
        """Create a sentence-like text from pool."""
        if max_words is None:
            max_words = random.randint(4, 10)
        
        words = []
        for _ in range(max_words):
            words.append(random.choice(text_pool))
        
        return ' '.join(words)
    
    def get_text_dimensions(self, draw, text, font):
        """Get text dimensions safely."""
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            width = max(bbox[2] - bbox[0], 1)
            height = max(bbox[3] - bbox[1], 1)
            ascent = max(-bbox[1], 1)
            return width, height, ascent
        except:
            return 1, 1, 1
    
    def generate_character_boxes(self, draw: ImageDraw, text: str, 
                                font: ImageFont, start_x: int, start_y: int) -> List[Dict]:
        """Generate character-level bounding boxes."""
        boxes = []
        current_x = start_x
        
        for char in text:
            if char.isspace():
                try:
                    char_width = draw.textbbox((current_x, start_y), char, font=font)[2] - current_x
                    current_x += max(char_width, 3)
                except:
                    current_x += 5
                continue
                
            try:
                bbox = draw.textbbox((current_x, start_y), char, font=font)
            except:
                continue
            
            if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                boxes.append({
                    'char': char,
                    'bbox': [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
                    'center': [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
                })
                
                char_width = bbox[2] - bbox[0]
                current_x += char_width
        
        return boxes
    
    def generate_ground_truth_maps(self, boxes: List[Dict], 
                                   img_width: int, img_height: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate region and affinity ground truth maps."""
        region_map = np.zeros((img_height, img_width), dtype=np.float32)
        affinity_map = np.zeros((img_height, img_width), dtype=np.float32)
        
        if not boxes:
            return region_map, affinity_map
        
        for box in boxes:
            x1, y1, x2, y2 = box['bbox']
            cx, cy = box['center']
            
            w = max(x2 - x1, 1)
            h = max(y2 - y1, 1)
            
            sigma_x = max(w / 2.5, 1.0)
            sigma_y = max(h / 2.5, 1.0)
            
            pad_x = int(w * 0.2)
            pad_y = int(h * 0.2)
            
            y_start = max(0, int(y1) - pad_y)
            y_end = min(img_height, int(y2) + pad_y)
            x_start = max(0, int(x1) - pad_x)
            x_end = min(img_width, int(x2) + pad_x)
            
            for y in range(y_start, y_end):
                for x in range(x_start, x_end):
                    gaussian_val = np.exp(-((x - cx)**2 / (2 * sigma_x**2) + 
                                           (y - cy)**2 / (2 * sigma_y**2)))
                    region_map[y, x] = max(region_map[y, x], gaussian_val)
        
        for i in range(len(boxes) - 1):
            box1 = boxes[i]
            box2 = boxes[i + 1]
            
            if abs(box1['center'][1] - box2['center'][1]) > 20:
                continue

            char_distance = abs(box2['center'][0] - box1['center'][0])
            avg_char_width = (box1['bbox'][2] - box1['bbox'][0] + box2['bbox'][2] - box2['bbox'][0]) / 2
            
            if char_distance > avg_char_width * 3:
                continue
            
            cx1, cy1 = box1['center']
            cx2, cy2 = box2['center']
            
            mid_x = (cx1 + cx2) / 2
            mid_y = (cy1 + cy2) / 2
            
            w = abs(cx2 - cx1)
            h = (box1['bbox'][3] - box1['bbox'][1] + box2['bbox'][3] - box2['bbox'][1]) / 2
            
            sigma_x = max(w / 2.0, 1.0)
            sigma_y = max(h / 2.5, 1.0)
            
            x_start = int(max(0, min(cx1, cx2) - w/2))
            x_end = int(min(img_width, max(cx1, cx2) + w/2))
            y_start = int(max(0, mid_y - h))
            y_end = int(min(img_height, mid_y + h))
            
            for y in range(y_start, y_end):
                for x in range(x_start, x_end):
                    gaussian_val = np.exp(-((x - mid_x)**2 / (2 * sigma_x**2) + 
                                           (y - mid_y)**2 / (2 * sigma_y**2)))
                    affinity_map[y, x] = max(affinity_map[y, x], gaussian_val)
        
        return region_map, affinity_map
    
    def apply_augmentation(self, img: Image.Image) -> Image.Image:
        """Apply random augmentations."""
        if random.random() > 0.5:
            brightness = random.uniform(0.7, 1.3)
            img = Image.eval(img, lambda x: int(min(255, max(0, x * brightness))))
        
        if random.random() > 0.5:
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(random.uniform(0.8, 1.2))
        
        if random.random() > 0.7:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
        
        if random.random() > 0.7:
            img_array = np.array(img)
            noise = np.random.normal(0, 5, img_array.shape)
            img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(img_array)
        
        return img
    
    def generate_multiline_document(self, draw, text_pool, font, all_boxes, layout_type='paragraph'):
        """Generate realistic multi-line document layouts with safety checks."""
        items = []
        
        # Calculate line metrics with safety checks
        try:
            test_bbox = draw.textbbox((0, 0), "ÁÂÃÄÅÆÇغ", font=font)
            line_height = max(test_bbox[3] - test_bbox[1], 20)  # Min 20px
            ascent = max(-test_bbox[1], 10)  # Min 10px
        except:
            line_height = 30
            ascent = 10
        
        line_spacing = random.randint(int(line_height * 0.2), int(line_height * 0.5))
        line_spacing = max(line_spacing, 5)  # Min 5px spacing
        
        # Calculate available space
        margin_top = random.randint(15, 30)
        margin_bottom = random.randint(15, 30)
        available_height = self.image_height - margin_top - margin_bottom
        
        # Calculate max lines with safety
        total_line_height = line_height + line_spacing
        if total_line_height <= 0:
            total_line_height = 40  # Fallback
        
        max_lines = max(1, int(available_height / total_line_height))
        
        # Adaptive line count based on image height
        if self.image_height <= 128:
            num_lines = 1
        elif self.image_height <= 256:
            num_lines = random.randint(2, min(4, max_lines))
        else:  # 512+
            num_lines = random.randint(5, min(12, max_lines))
        
        current_y = margin_top + ascent
        
        # Ensure we don't start outside bounds
        if current_y < 10:
            current_y = 10
        if current_y > self.image_height - 50:
            current_y = 30
        
        if layout_type == 'paragraph':
            for line_num in range(num_lines):
                if current_y + line_height > self.image_height - margin_bottom:
                    break
                
                text = self.create_sentence(text_pool, max_words=random.randint(6, 14))
                
                if not self.is_text_supported(font, text):
                    continue
                
                try:
                    text_width, _, _ = self.get_text_dimensions(draw, text, font)
                    
                    words = text.split()
                    while text_width > self.image_width - 60 and len(words) > 2:
                        words.pop()
                        text = ' '.join(words)
                        text_width, _, _ = self.get_text_dimensions(draw, text, font)
                    
                    x_pos = random.randint(20, 40)
                    if x_pos + text_width > self.image_width - 20:
                        x_pos = 20
                    
                    text_color = (random.randint(0, 60), random.randint(0, 60), random.randint(0, 60))
                    draw.text((x_pos, current_y), text, font=font, fill=text_color)
                    
                    boxes = self.generate_character_boxes(draw, text, font, x_pos, current_y)
                    all_boxes.extend(boxes)
                    items.append(text)
                    
                    current_y += line_height + line_spacing
                except:
                    continue
        
        elif layout_type == 'invoice_line':
            for line_num in range(num_lines):
                if current_y + line_height > self.image_height - margin_bottom:
                    break
                
                item_name = self.create_sentence(text_pool, max_words=random.randint(2, 5))
                
                if not self.is_text_supported(font, item_name):
                    continue
                
                try:
                    x_left = random.randint(20, 30)
                    text_color = (random.randint(0, 60), random.randint(0, 60), random.randint(0, 60))
                    draw.text((x_left, current_y), item_name, font=font, fill=text_color)
                    
                    boxes = self.generate_character_boxes(draw, item_name, font, x_left, current_y)
                    all_boxes.extend(boxes)
                    
                    amount = random.choice(text_pool)
                    if self.is_text_supported(font, amount):
                        amount_width, _, _ = self.get_text_dimensions(draw, amount, font)
                        x_right = self.image_width - amount_width - random.randint(20, 30)
                        
                        item_width, _, _ = self.get_text_dimensions(draw, item_name, font)
                        
                        if x_right > x_left + item_width + 40 and x_right > 0:
                            draw.text((x_right, current_y), amount, font=font, fill=text_color)
                            boxes = self.generate_character_boxes(draw, amount, font, x_right, current_y)
                            all_boxes.extend(boxes)
                            items.append(f"{item_name} ... {amount}")
                        else:
                            items.append(item_name)
                    else:
                        items.append(item_name)
                    
                    current_y += line_height + line_spacing
                except:
                    continue
        
        elif layout_type == 'form_field':
            for line_num in range(num_lines):
                if current_y + line_height > self.image_height - margin_bottom:
                    break
                
                label = random.choice(text_pool)
                value = self.create_sentence(text_pool, max_words=random.randint(2, 4))
                
                if not self.is_text_supported(font, label):
                    continue
                
                try:
                    x_pos = random.randint(20, 40)
                    
                    text_color = (random.randint(0, 50), random.randint(0, 50), random.randint(0, 50))
                    draw.text((x_pos, current_y), label + ":", font=font, fill=text_color)
                    
                    boxes = self.generate_character_boxes(draw, label + ":", font, x_pos, current_y)
                    all_boxes.extend(boxes)
                    
                    label_width, _, _ = self.get_text_dimensions(draw, label + ":", font)
                    x_value = x_pos + label_width + random.randint(10, 20)
                    
                    if self.is_text_supported(font, value) and x_value < self.image_width - 50:
                        value_width, _, _ = self.get_text_dimensions(draw, value, font)
                        
                        words = value.split()
                        while value_width > self.image_width - x_value - 20 and len(words) > 1:
                            words.pop()
                            value = ' '.join(words)
                            value_width, _, _ = self.get_text_dimensions(draw, value, font)
                        
                        if x_value + value_width <= self.image_width - 20:
                            value_color = (random.randint(20, 80), random.randint(20, 80), random.randint(20, 80))
                            draw.text((x_value, current_y), value, font=font, fill=value_color)
                            
                            boxes = self.generate_character_boxes(draw, value, font, x_value, current_y)
                            all_boxes.extend(boxes)
                            items.append(f"{label}: {value}")
                        else:
                            items.append(label + ":")
                    else:
                        items.append(label + ":")
                    
                    current_y += line_height + line_spacing
                except:
                    continue
        
        elif layout_type == 'table_row':
            num_cols = random.randint(2, 4)
            col_width = self.image_width // num_cols
            
            for line_num in range(num_lines):
                if current_y + line_height > self.image_height - margin_bottom:
                    break
                
                row_items = []
                for col in range(num_cols):
                    cell_text = random.choice(text_pool)
                    
                    if not self.is_text_supported(font, cell_text):
                        continue
                    
                    try:
                        text_width, _, _ = self.get_text_dimensions(draw, cell_text, font)
                        x_pos = col * col_width + random.randint(10, 15)
                        
                        if x_pos + text_width <= (col + 1) * col_width - 10 and x_pos >= 0:
                            text_color = (random.randint(0, 70), random.randint(0, 70), random.randint(0, 70))
                            draw.text((x_pos, current_y), cell_text, font=font, fill=text_color)
                            
                            boxes = self.generate_character_boxes(draw, cell_text, font, x_pos, current_y)
                            all_boxes.extend(boxes)
                            row_items.append(cell_text)
                    except:
                        continue
                
                if row_items:
                    items.append(' | '.join(row_items))
                
                current_y += line_height + line_spacing
        
        elif layout_type == 'list_items':
            # Bullet point list
            for line_num in range(num_lines):
                if current_y + line_height > self.image_height - margin_bottom:
                    break
                
                text = self.create_sentence(text_pool, max_words=random.randint(4, 8))
                
                if not self.is_text_supported(font, text):
                    continue
                
                try:
                    # Add bullet
                    x_bullet = random.randint(25, 35)
                    bullet = "•"
                    text_color = (random.randint(0, 60), random.randint(0, 60), random.randint(0, 60))
                    draw.text((x_bullet, current_y), bullet, font=font, fill=text_color)
                    
                    # Add text after bullet
                    x_text = x_bullet + 20
                    
                    text_width, _, _ = self.get_text_dimensions(draw, text, font)
                    words = text.split()
                    while text_width > self.image_width - x_text - 20 and len(words) > 2:
                        words.pop()
                        text = ' '.join(words)
                        text_width, _, _ = self.get_text_dimensions(draw, text, font)
                    
                    draw.text((x_text, current_y), text, font=font, fill=text_color)
                    
                    boxes = self.generate_character_boxes(draw, bullet + " " + text, font, x_bullet, current_y)
                    all_boxes.extend(boxes)
                    items.append(f"• {text}")
                    
                    current_y += line_height + line_spacing
                except:
                    continue
        
        elif layout_type == 'header':
            # Header + body
            text = self.create_sentence(text_pool, max_words=random.randint(2, 4))
            
            if self.is_text_supported(font, text):
                try:
                    text_width, _, _ = self.get_text_dimensions(draw, text, font)
                    x_pos = (self.image_width - text_width) // 2
                    x_pos = max(20, min(x_pos, self.image_width - text_width - 20))
                    
                    text_color = (random.randint(0, 40), random.randint(0, 40), random.randint(0, 40))
                    draw.text((x_pos, current_y), text, font=font, fill=text_color)
                    
                    boxes = self.generate_character_boxes(draw, text, font, x_pos, current_y)
                    all_boxes.extend(boxes)
                    items.append(text)
                    
                    current_y += line_height + line_spacing + 10
                except:
                    pass
            
            # Body lines
            for line_num in range(num_lines - 1):
                if current_y + line_height > self.image_height - margin_bottom:
                    break
                
                text = self.create_sentence(text_pool, max_words=random.randint(5, 10))
                
                if not self.is_text_supported(font, text):
                    continue
                
                try:
                    text_width, _, _ = self.get_text_dimensions(draw, text, font)
                    
                    words = text.split()
                    while text_width > self.image_width - 60 and len(words) > 2:
                        words.pop()
                        text = ' '.join(words)
                        text_width, _, _ = self.get_text_dimensions(draw, text, font)
                    
                    x_pos = random.randint(25, 35)
                    
                    text_color = (random.randint(0, 60), random.randint(0, 60), random.randint(0, 60))
                    draw.text((x_pos, current_y), text, font=font, fill=text_color)
                    
                    boxes = self.generate_character_boxes(draw, text, font, x_pos, current_y)
                    all_boxes.extend(boxes)
                    items.append(text)
                    
                    current_y += line_height + line_spacing
                except:
                    continue
        
        return items
    
    def generate_sample(self, text_pool: List[str], font_path: str, font_size: int, 
                       idx: int, language: str = 'unknown', augment: bool = True) -> Dict:
        """Generate a realistic multi-line document sample."""
        bg_color = random.randint(240, 255)
        img = Image.new('RGB', (self.image_width, self.image_height), 
                       color=(bg_color, bg_color, bg_color))
        draw = ImageDraw.Draw(img)
        
        try:
            font = ImageFont.truetype(font_path, font_size)
        except Exception as e:
            return None
        
        # Test if font actually works with this text
        sample_text = ' '.join(text_pool[:5])
        if not self.test_font_support(font, sample_text):
            return None
        
        all_boxes = []
        full_text = []
        
        # More layout variety
        layout_types = ['paragraph', 'invoice_line', 'form_field', 'table_row', 'list_items', 'header']
        layout_weights = [0.30, 0.25, 0.20, 0.12, 0.08, 0.05]
        layout = random.choices(layout_types, weights=layout_weights)[0]
        
        items = self.generate_multiline_document(draw, text_pool, font, all_boxes, layout)
        full_text = items
        
        if not all_boxes:
            return None
            
        region_map, affinity_map = self.generate_ground_truth_maps(
            all_boxes, self.image_width, self.image_height
        )
        
        if augment:
            img = self.apply_augmentation(img)
        
        img_filename = f'img_{idx:06d}.jpg'
        img_path = os.path.join(self.output_dir, 'images', img_filename)
        img.save(img_path, quality=95)
        
        region_filename = f'region_{idx:06d}.npy'
        affinity_filename = f'affinity_{idx:06d}.npy'
        
        np.save(os.path.join(self.output_dir, 'labels', region_filename), region_map)
        np.save(os.path.join(self.output_dir, 'labels', affinity_filename), affinity_map)
        
        annotation = {
            'image': img_filename,
            'text': "\n".join(full_text) if full_text else "",
            'layout': layout,
            'num_lines': len(full_text),
            'language': language,
            'font': os.path.basename(font_path),
            'font_size': font_size,
            'num_chars': len(all_boxes),
            'boxes': all_boxes,
            'region_map': region_filename,
            'affinity_map': affinity_filename
        }
        
        annotation_path = os.path.join(self.output_dir, 'annotations', f'anno_{idx:06d}.json')
        with open(annotation_path, 'w', encoding='utf-8') as f:
            json.dump(annotation, f, ensure_ascii=False, indent=2)
        
        return annotation
    
    def generate_dataset(self, text_files: Union[str, Dict[str, str]], 
                        font_dir: Union[str, Dict[str, str]], 
                        num_samples: int = 1000,
                        font_size_range: Tuple[int, int] = (18, 32),
                        augment: bool = True,
                        language_ratio: Dict[str, float] = None) -> None:
        """Generate complete dataset."""
        print("\n" + "="*60)
        print("  CRAFT Dataset Generator (Multi-Line Real Documents)")
        print("="*60)
        
        if isinstance(text_files, dict):
            text_data = {}
            for lang, file in text_files.items():
                lines = self.load_text_lines(file)
                if lines:
                    text_data[lang] = lines
                    print(f"✓ Loaded {len(lines)} {lang} text lines")
                else:
                    print(f"⚠ Warning: No text loaded for {lang}")
            
            if not text_data:
                print("❌ Error: No text data loaded!")
                return
                
            if language_ratio is None:
                language_ratio = {lang: 1.0/len(text_data) for lang in text_data}
        else:
            text_data = {'default': self.load_text_lines(text_files)}
            language_ratio = {'default': 1.0}
            if text_data['default']:
                print(f"✓ Loaded {len(text_data['default'])} text lines")
            else:
                print("❌ Error: No text data loaded!")
                return
        
        if isinstance(font_dir, dict):
            fonts_data = {}
            for lang, dir_path in font_dir.items():
                fonts = self.get_font_list(dir_path)
                if fonts:
                    fonts_data[lang] = fonts
                    print(f"✓ Found {len(fonts)} {lang} fonts")
                else:
                    print(f"⚠ Warning: No fonts found for {lang}")
        else:
            fonts_data = {'default': self.get_font_list(font_dir)}
            if fonts_data['default']:
                print(f"✓ Found {len(fonts_data['default'])} fonts")
            else:
                print("❌ Error: No fonts found!")
                return
        
        if not any(fonts_data.values()):
            print("❌ Error: No fonts available!")
            return
        
        print(f"\nGenerating {num_samples} samples...")
        print(f"Image size: {self.image_width}x{self.image_height}")
        print(f"Font size range: {font_size_range[0]}-{font_size_range[1]} (smaller = more realistic)")
        
        if self.image_height <= 128:
            print(f"Lines per image: 1 (height={self.image_height})")
        elif self.image_height <= 256:
            print(f"Lines per image: 2-4 (height={self.image_height})")
        else:
            print(f"Lines per image: 5-12 (height={self.image_height})")
        
        print(f"Layouts: Paragraph (30%), Invoice (25%), Form (20%), Table (12%), List (8%), Header (5%)")
        print(f"Augmentation: {'Enabled' if augment else 'Disabled'}")
        if len(language_ratio) > 1:
            print(f"Language ratio: {language_ratio}")
        print("="*60 + "\n")
        
        dataset_info = []
        successful = 0
        layout_stats = {}
        total_lines = 0
        skipped_fonts = 0
        
        for i in range(num_samples):
            lang = random.choices(list(language_ratio.keys()), 
                                weights=list(language_ratio.values()))[0]
            
            if lang not in text_data or not text_data[lang]:
                continue
            text_pool = text_data[lang]
            
            font_key = lang if lang in fonts_data else 'default'
            if font_key not in fonts_data or not fonts_data[font_key]:
                continue
            font_path = random.choice(fonts_data[font_key])
            
            # Weighted font sizes: favor smaller fonts (more realistic)
            # 60% small (18-24), 30% medium (25-28), 10% large (29-32)
            rand_val = random.random()
            if rand_val < 0.6:
                font_size = random.randint(18, 24)
            elif rand_val < 0.9:
                font_size = random.randint(25, 28)
            else:
                font_size = random.randint(29, 32)
            
            annotation = self.generate_sample(text_pool, font_path, font_size, i, lang, augment)
            
            if annotation:
                dataset_info.append(annotation)
                successful += 1
                total_lines += annotation.get('num_lines', 0)
                
                layout = annotation.get('layout', 'unknown')
                layout_stats[layout] = layout_stats.get(layout, 0) + 1
            else:
                skipped_fonts += 1
            
            if (i + 1) % 100 == 0:
                print(f"Progress: {i + 1}/{num_samples} ({successful} successful, {skipped_fonts} skipped)")
        
        lang_counts = {}
        for item in dataset_info:
            lang = item.get('language', 'unknown')
            lang_counts[lang] = lang_counts.get(lang, 0) + 1
        
        dataset_summary = {
            'num_samples': len(dataset_info),
            'successful_samples': successful,
            'total_lines': total_lines,
            'avg_lines_per_image': total_lines / successful if successful > 0 else 0,
            'image_size': [self.image_width, self.image_height],
            'language_distribution': lang_counts,
            'layout_distribution': layout_stats,
            'font_size_range': font_size_range,
            'augmentation_enabled': augment
        }
        
        with open(os.path.join(self.output_dir, 'dataset_info.json'), 'w', encoding='utf-8') as f:
            json.dump(dataset_summary, f, ensure_ascii=False, indent=2)
        
        with open(os.path.join(self.output_dir, 'annotations_list.json'), 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, ensure_ascii=False, indent=2)
        
        print(f"\n{'='*60}")
        print(f"✅ Dataset generation complete!")
        print(f"{'='*60}")
        print(f"  Total samples: {len(dataset_info)}/{num_samples}")
        print(f"  Success rate: {successful/num_samples*100:.1f}%")
        print(f"  Skipped (bad fonts): {skipped_fonts}")
        print(f"  Total lines: {total_lines}")
        print(f"  Avg lines/image: {total_lines/successful:.1f}")
        print(f"  Layout distribution: {layout_stats}")
        print(f"  Output directory: {self.output_dir}")
        print(f"{'='*60}\n")


def generate_detector_dataset_command(args):
    """CLI Command handler for dataset generation."""
    text_files = args.text_file
    font_dirs = args.fonts_dir
    language_ratio = None
    
    if isinstance(args.text_file, str) and ':' in args.text_file and not os.path.exists(args.text_file):
        text_files = {}
        for item in args.text_file.split(','):
            if ':' in item:
                lang, file = item.split(':', 1)
                text_files[lang.strip()] = file.strip()
    
    if isinstance(args.fonts_dir, str) and ':' in args.fonts_dir and not os.path.exists(args.fonts_dir):
        font_dirs = {}
        for item in args.fonts_dir.split(','):
            if ':' in item:
                lang, dir_path = item.split(':', 1)
                font_dirs[lang.strip()] = dir_path.strip()
    
    if hasattr(args, 'language_ratio') and args.language_ratio:
        language_ratio = {}
        for item in args.language_ratio.split(','):
            lang, ratio = item.split(':')
            language_ratio[lang.strip()] = float(ratio.strip())
    
    image_height = getattr(args, 'image_height', 512)
    
    generator = MultilingualDatasetGenerator(
        output_dir=args.output,
        image_width=512,
        image_height=image_height
    )
    
    num_samples = args.num_train + args.num_val
    
    # Smaller font sizes (18-32 instead of 24-40) - more realistic
    generator.generate_dataset(
        text_files=text_files,
        font_dir=font_dirs,
        num_samples=num_samples,
        font_size_range=(18, 32),
        augment=not args.no_augment,
        language_ratio=language_ratio
    )
