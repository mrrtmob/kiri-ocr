"""
Multilingual Dataset Generator for Text Detection (CRAFT-style)
Supports Khmer, English, and other languages
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import os
import json
import random
from typing import Union, Dict, List, Tuple
from pathlib import Path

class MultilingualDatasetGenerator:
    """Dataset generator supporting multiple languages (Khmer, English, etc.)"""
    
    def __init__(self, output_dir='dataset', image_width=512, image_height=64):
        self.output_dir = output_dir
        self.image_width = image_width
        self.image_height = image_height
        
        # Create output directories
        os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'annotations'), exist_ok=True)
        
    def load_text_lines(self, text_file: Union[str, List[str]]) -> List[str]:
        """Load text lines from file(s)"""
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
        """Get list of font files"""
        font_files = []
        if not os.path.isdir(font_dir):
            print(f"Warning: Font directory '{font_dir}' not found")
            return font_files
            
        for file in os.listdir(font_dir):
            if file.endswith(('.ttf', '.otf', '.TTF', '.OTF')):
                font_files.append(os.path.join(font_dir, file))
        return font_files
    
    def generate_character_boxes(self, draw: ImageDraw, text: str, 
                                font: ImageFont, start_x: int, start_y: int) -> List[Dict]:
        """Generate character-level bounding boxes"""
        boxes = []
        current_x = start_x
        
        for char in text:
            if char.isspace():  # Skip spaces but advance position
                try:
                    char_width = draw.textbbox((current_x, start_y), char, font=font)[2] - current_x
                    current_x += char_width
                except:
                    current_x += 5  # Default space width
                continue
                
            # Get bounding box for character
            try:
                bbox = draw.textbbox((current_x, start_y), char, font=font)
            except:
                continue
            
            if bbox[2] > bbox[0] and bbox[3] > bbox[1]:  # Valid box
                boxes.append({
                    'char': char,
                    'bbox': [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
                    'center': [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
                })
                
                # Move to next character position
                char_width = bbox[2] - bbox[0]
                current_x += char_width
        
        return boxes
    
    def generate_ground_truth_maps(self, boxes: List[Dict], 
                                   img_width: int, img_height: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate region and affinity ground truth maps"""
        region_map = np.zeros((img_height, img_width), dtype=np.float32)
        affinity_map = np.zeros((img_height, img_width), dtype=np.float32)
        
        if not boxes:
            return region_map, affinity_map
        
        # Generate region map (Gaussian heatmap for each character)
        for box in boxes:
            x1, y1, x2, y2 = box['bbox']
            cx, cy = box['center']
            
            # Character region size
            w = max(x2 - x1, 1)
            h = max(y2 - y1, 1)
            
            # Generate Gaussian heatmap
            sigma_x = max(w / 3.0, 0.5)
            sigma_y = max(h / 3.0, 0.5)
            
            # Expand region slightly to ensure coverage
            y_start = max(0, int(y1) - 2)
            y_end = min(img_height, int(y2) + 3)
            x_start = max(0, int(x1) - 2)
            x_end = min(img_width, int(x2) + 3)
            
            for y in range(y_start, y_end):
                for x in range(x_start, x_end):
                    # Gaussian formula
                    gaussian_val = np.exp(-((x - cx)**2 / (2 * sigma_x**2) + 
                                           (y - cy)**2 / (2 * sigma_y**2)))
                    region_map[y, x] = max(region_map[y, x], gaussian_val)
        
        # Generate affinity map (connections between characters)
        for i in range(len(boxes) - 1):
            box1 = boxes[i]
            box2 = boxes[i + 1]
            
            # Center points of two consecutive characters
            cx1, cy1 = box1['center']
            cx2, cy2 = box2['center']
            
            # Skip if characters are too far apart (likely different words)
            char_distance = abs(cx2 - cx1)
            avg_char_width = (box1['bbox'][2] - box1['bbox'][0] + box2['bbox'][2] - box2['bbox'][0]) / 2
            if char_distance > avg_char_width * 3:  # Threshold for word boundary
                continue
            
            # Midpoint between characters
            mid_x = (cx1 + cx2) / 2
            mid_y = (cy1 + cy2) / 2
            
            # Draw affinity region
            w = abs(cx2 - cx1)
            h = (box1['bbox'][3] - box1['bbox'][1] + box2['bbox'][3] - box2['bbox'][1]) / 2
            
            sigma_x = max(w / 2.0, 0.5)
            sigma_y = max(h / 3.0, 0.5)
            
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
        """Apply random augmentations"""
        # Random brightness
        if random.random() > 0.5:
            brightness = random.uniform(0.7, 1.3)
            img = Image.eval(img, lambda x: int(min(255, max(0, x * brightness))))
        
        # Random contrast
        if random.random() > 0.5:
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(random.uniform(0.8, 1.2))
        
        # Random blur
        if random.random() > 0.7:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
        
        # Random noise
        if random.random() > 0.7:
            img_array = np.array(img)
            noise = np.random.normal(0, 5, img_array.shape)
            img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(img_array)
        
        return img
    
    def generate_sample(self, text: str, font_path: str, font_size: int, 
                       idx: int, language: str = 'unknown', augment: bool = True) -> Dict:
        """Generate a single training sample"""
        # Create blank image with random background
        bg_color = random.randint(240, 255)
        img = Image.new('RGB', (self.image_width, self.image_height), 
                       color=(bg_color, bg_color, bg_color))
        draw = ImageDraw.Draw(img)
        
        # Load font
        try:
            font = ImageFont.truetype(font_path, font_size)
        except Exception as e:
            # print(f"Failed to load font: {font_path} - {e}")
            return None
        
        try:
            # Calculate text position
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # Check if text fits
            if text_width > self.image_width - 20:
                # Text too long, skip
                return None
            
            # Random horizontal position with margins
            margin = 10
            if text_width < self.image_width - 2 * margin:
                start_x = random.randint(margin, self.image_width - text_width - margin)
            else:
                start_x = margin
            
            # Center vertically
            start_y = (self.image_height - text_height) // 2
            
            # Random text color (darker colors for visibility)
            text_color = (
                random.randint(0, 80),
                random.randint(0, 80),
                random.randint(0, 80)
            )
            
            # Draw text
            draw.text((start_x, start_y), text, font=font, fill=text_color)
            
            # Generate character boxes
            boxes = self.generate_character_boxes(draw, text, font, start_x, start_y)
            
            if not boxes:
                return None
            
            # Generate ground truth maps
            region_map, affinity_map = self.generate_ground_truth_maps(
                boxes, self.image_width, self.image_height
            )
            
            # Apply augmentation
            if augment:
                img = self.apply_augmentation(img)
                
        except Exception as e:
            # print(f"Error processing sample {idx}: {e}")
            return None
        
        # Save image
        img_filename = f'img_{idx:06d}.jpg'
        img_path = os.path.join(self.output_dir, 'images', img_filename)
        img.save(img_path, quality=95)
        
        # Save ground truth maps
        region_filename = f'region_{idx:06d}.npy'
        affinity_filename = f'affinity_{idx:06d}.npy'
        
        np.save(os.path.join(self.output_dir, 'labels', region_filename), region_map)
        np.save(os.path.join(self.output_dir, 'labels', affinity_filename), affinity_map)
        
        # Save annotation
        annotation = {
            'image': img_filename,
            'text': text,
            'language': language,
            'font': os.path.basename(font_path),
            'font_size': font_size,
            'num_chars': len(boxes),
            'boxes': boxes,
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
                        font_size_range: Tuple[int, int] = (20, 48),
                        augment: bool = True,
                        language_ratio: Dict[str, float] = None) -> None:
        """Generate complete dataset with multi-language support
        
        Args:
            text_files: Single file or dict like {'khmer': 'khmer.txt', 'english': 'english.txt'}
            font_dir: Single dir or dict like {'khmer': 'fonts/khmer/', 'english': 'fonts/english/'}
            num_samples: Total samples to generate
            font_size_range: (min_size, max_size)
            augment: Apply augmentations
            language_ratio: Dict like {'khmer': 0.7, 'english': 0.3}
        """
        print("\n" + "="*60)
        print("  CRAFT Dataset Generator")
        print("="*60)
        
        # Handle text files
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
                
            # Set default ratio
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
        
        # Handle font directories
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
        print(f"Font size range: {font_size_range[0]}-{font_size_range[1]}")
        print(f"Augmentation: {'Enabled' if augment else 'Disabled'}")
        if len(language_ratio) > 1:
            print(f"Language ratio: {language_ratio}")
        print("="*60 + "\n")
        
        # Generate samples
        dataset_info = []
        successful = 0
        
        for i in range(num_samples):
            # Select language
            lang = random.choices(list(language_ratio.keys()), 
                                weights=list(language_ratio.values()))[0]
            
            # Get text
            if lang not in text_data or not text_data[lang]:
                continue
            text = random.choice(text_data[lang])
            
            # Get font
            font_key = lang if lang in fonts_data else 'default'
            if font_key not in fonts_data or not fonts_data[font_key]:
                continue
            font_path = random.choice(fonts_data[font_key])
            
            # Random font size
            font_size = random.randint(font_size_range[0], font_size_range[1])
            
            # Generate sample
            annotation = self.generate_sample(text, font_path, font_size, i, lang, augment)
            
            if annotation:
                dataset_info.append(annotation)
                successful += 1
            
            if (i + 1) % 100 == 0:
                print(f"Progress: {i + 1}/{num_samples} ({successful} successful)")
        
        # Calculate statistics
        lang_counts = {}
        for item in dataset_info:
            lang = item.get('language', 'unknown')
            lang_counts[lang] = lang_counts.get(lang, 0) + 1
        
        # Save dataset info
        dataset_summary = {
            'num_samples': len(dataset_info),
            'successful_samples': successful,
            'image_size': [self.image_width, self.image_height],
            'language_distribution': lang_counts,
            'font_size_range': font_size_range,
            'augmentation_enabled': augment
        }
        
        with open(os.path.join(self.output_dir, 'dataset_info.json'), 'w', encoding='utf-8') as f:
            json.dump(dataset_summary, f, ensure_ascii=False, indent=2)
        
        # Save detailed annotations separately
        with open(os.path.join(self.output_dir, 'annotations_list.json'), 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, ensure_ascii=False, indent=2)
        
        print(f"\n{'='*60}")
        print(f"✅ Dataset generation complete!")
        print(f"{'='*60}")
        print(f"  Total samples: {len(dataset_info)}/{num_samples}")
        print(f"  Success rate: {successful/num_samples*100:.1f}%")
        if len(lang_counts) > 1:
            print(f"  Language distribution:")
            for lang, count in lang_counts.items():
                print(f"    - {lang}: {count} ({count/len(dataset_info)*100:.1f}%)")
        print(f"  Output directory: {self.output_dir}")
        print(f"{'='*60}\n")


def generate_detector_dataset_command(args):
    """CLI Command handler for kiri-ocr"""
    
    # Parse multi-language arguments if provided
    text_files = args.text_file
    font_dirs = args.fonts_dir
    language_ratio = None
    
    # Check for multi-language format (lang:file)
    if isinstance(args.text_file, str) and ':' in args.text_file and not os.path.exists(args.text_file):
        # Multi-language mode
        text_files = {}
        for item in args.text_file.split(','):
            if ':' in item:
                lang, file = item.split(':', 1)
                text_files[lang.strip()] = file.strip()
    
    # Check for multi-language fonts (lang:dir)
    if isinstance(args.fonts_dir, str) and ':' in args.fonts_dir and not os.path.exists(args.fonts_dir):
        font_dirs = {}
        for item in args.fonts_dir.split(','):
            if ':' in item:
                lang, dir_path = item.split(':', 1)
                font_dirs[lang.strip()] = dir_path.strip()
    
    # Parse language ratio if provided (e.g., "khmer:0.7,english:0.3")
    if hasattr(args, 'language_ratio') and args.language_ratio:
        language_ratio = {}
        for item in args.language_ratio.split(','):
            lang, ratio = item.split(':')
            language_ratio[lang.strip()] = float(ratio.strip())
    
    # Initialize generator
    generator = MultilingualDatasetGenerator(
        output_dir=args.output,
        image_width=512,
        image_height=64
    )
    
    # Calculate total samples
    num_samples = args.num_train + args.num_val
    
    # Generate dataset
    generator.generate_dataset(
        text_files=text_files,
        font_dir=font_dirs,
        num_samples=num_samples,
        font_size_range=(24, 48),
        augment=not args.no_augment,
        language_ratio=language_ratio
    )