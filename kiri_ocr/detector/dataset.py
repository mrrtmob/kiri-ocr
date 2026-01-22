import os
import random
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import argparse
from tqdm import tqdm

try:
    from ..generator import FontManager
except ImportError:
    # Fallback if generator not found (e.g. running standalone)
    # This might fail if relative import fails, so try absolute or assume installed
    try:
        from kiri_ocr.generator import FontManager
    except ImportError:
        FontManager = None

class DetectorDatasetGenerator:
    def __init__(self, output_dir='detector_dataset', fonts_dir='fonts'):
        self.output_dir = output_dir
        self.fonts_dir = fonts_dir
        self.font_manager = None
        if FontManager:
            # We use FontManager to find fonts, but we'll use paths to load with random sizes
            self.font_manager = FontManager(fonts_dir=fonts_dir)
        
        self.setup_directories()
        
    def setup_directories(self):
        """Create dataset directory structure"""
        dirs = [
            f'{self.output_dir}/images/train',
            f'{self.output_dir}/images/val',
            f'{self.output_dir}/labels/train',
            f'{self.output_dir}/labels/val'
        ]
        for d in dirs:
            os.makedirs(d, exist_ok=True)
    
    def load_khmer_text(self, text_file):
        """Load Khmer text from file"""
        with open(text_file, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        return lines
    
    def get_random_background_color(self):
        """Generate random background color"""
        colors = [
            (255, 255, 255),  # White
            (240, 240, 240),  # Light gray
            (255, 250, 240),  # Floral white
            (245, 245, 220),  # Beige
            (230, 230, 250),  # Lavender
            (250, 250, 250),  # Very light gray
            (255, 255, 240),  # Ivory
        ]
        return random.choice(colors)
    
    def get_random_text_color(self):
        """Generate random text color"""
        colors = [
            (0, 0, 0),        # Black
            (50, 50, 50),     # Dark gray
            (0, 0, 139),      # Dark blue
            (139, 0, 0),      # Dark red
            (0, 100, 0),      # Dark green
            (30, 30, 30),     # Almost black
        ]
        return random.choice(colors)
    
    def _is_text_supported(self, font, text):
        """Check if font supports the characters in text (detects 'tofu' boxes)"""
        try:
            # Get the glyph for a strictly undefined character to use as reference
            undefined_chars = ['\uFFFF', '\U0010FFFF', '\0']
            ref_mask = None
            ref_bbox = None
            
            for uc in undefined_chars:
                try:
                    ref_mask = font.getmask(uc)
                    ref_bbox = ref_mask.getbbox()
                    if ref_mask:
                        break
                except Exception:
                    continue
            
            if ref_mask is None:
                # Can't determine reference, assume supported
                return True

            ref_bytes = bytes(ref_mask)

            for char in text:
                # Skip spaces and control characters
                if char.isspace() or ord(char) < 32:
                    continue
                    
                try:
                    char_mask = font.getmask(char)
                    char_bbox = char_mask.getbbox()
                    
                    # Compare with reference "notdef" glyph
                    if char_bbox == ref_bbox:
                        # Exact bbox match. Deep check bytes.
                        if bytes(char_mask) == ref_bytes:
                            # It's a tofu/box!
                            return False
                except Exception:
                    return False
                    
            return True
        except Exception:
            return True
    
    def apply_augmentation(self, image):
        """Apply random augmentations to the full page image"""
        img_array = np.array(image)
        
        # 1. Noise (Gaussian)
        if random.random() > 0.5:
            noise = np.random.normal(0, random.randint(5, 20), img_array.shape)
            img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
            
        # 2. Blur
        if random.random() > 0.6:
            ksize = random.choice([3, 5])
            img_array = cv2.GaussianBlur(img_array, (ksize, ksize), 0)
            
        # 3. Brightness/Contrast
        if random.random() > 0.5:
            alpha = random.uniform(0.8, 1.2) # Contrast
            beta = random.randint(-30, 30)   # Brightness
            img_array = cv2.convertScaleAbs(img_array, alpha=alpha, beta=beta)
            
        # 4. JPEG Compression artifacts
        if random.random() > 0.5:
            quality = random.randint(50, 95)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            result, encimg = cv2.imencode('.jpg', img_array, encode_param)
            img_array = cv2.imdecode(encimg, 1)

        return Image.fromarray(img_array)
    
    def generate_single_image(self, texts, image_idx, split='train', 
                              min_lines=3, max_lines=10, augment=True, 
                              specific_font_path=None):
        """Generate a single image with multiple text regions"""
        
        # Image dimensions
        img_width = random.randint(800, 1200)
        img_height = random.randint(600, 1000)
        
        # Create image
        bg_color = self.get_random_background_color()
        image = Image.new('RGB', (img_width, img_height), bg_color)
        draw = ImageDraw.Draw(image)
        
        # Determine number of lines
        num_lines = random.randint(min_lines, min(max_lines, len(texts)))
        selected_texts = random.sample(texts, num_lines)
        
        # Annotations (YOLO format)
        annotations = []
        
        # Layout strategy
        y_offset = random.randint(50, 100)
        x_margin = random.randint(50, 100)
        line_spacing = random.randint(10, 40)
        
        # Common text color for the page
        text_color = self.get_random_text_color()
        
        # Extract unique font paths for random selection
        available_font_paths = []
        if self.font_manager and self.font_manager.all_fonts:
            available_font_paths = list(set([f[0] for f in self.font_manager.all_fonts]))
        
        for text in selected_texts:
            font_size = random.randint(24, 48)
            font = None
            
            # Try specific font first
            if specific_font_path:
                try:
                    candidate = ImageFont.truetype(specific_font_path, font_size)
                    if self._is_text_supported(candidate, text):
                        font = candidate
                except:
                    pass
            
            # Random font selection with retry
            if font is None and available_font_paths:
                retries = 10
                for _ in range(retries):
                    random_path = random.choice(available_font_paths)
                    try:
                        candidate = ImageFont.truetype(random_path, font_size)
                        if self._is_text_supported(candidate, text):
                            font = candidate
                            break
                    except:
                        continue
            
            # Fallback
            if font is None:
                 font = ImageFont.load_default()

            # Get text bounding box (with error handling)
            try:
                try:
                    bbox = draw.textbbox((0, 0), text, font=font)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                except AttributeError:
                    # Fallback for older Pillow
                    text_width, text_height = draw.textsize(text, font=font)
                    bbox = (0, 0, text_width, text_height)
            except Exception:
                # Skip line if measurement fails
                continue
            
            # Layout
            align = random.choice(['left', 'center', 'random'])
            if align == 'left':
                x_offset = x_margin + random.randint(0, 20)
            elif align == 'center':
                x_offset = (img_width - text_width) // 2
            else:
                x_offset = random.randint(x_margin, max(x_margin + 1, img_width - text_width - x_margin))
            
            # Boundary checks
            x_offset = max(10, min(x_offset, img_width - text_width - 10))
            if y_offset + text_height > img_height - 50:
                break
            
            # Draw
            draw.text((x_offset, y_offset), text, font=font, fill=text_color)
            
            # Calculate bounding box
            x1 = x_offset
            y1 = y_offset
            x2 = x_offset + text_width
            y2 = y_offset + text_height
            
            # Padding
            padding = random.randint(2, 8)
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(img_width, x2 + padding)
            y2 = min(img_height, y2 + padding)
            
            # YOLO format
            x_center = ((x1 + x2) / 2) / img_width
            y_center = ((y1 + y2) / 2) / img_height
            width = (x2 - x1) / img_width
            height = (y2 - y1) / img_height
            
            annotations.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
            y_offset += text_height + line_spacing
        
        # Augmentation
        if augment:
            image = self.apply_augmentation(image)
        
        # Rotation
        if augment and random.random() > 0.7:
            angle = random.uniform(-2, 2)
            image = image.rotate(angle, expand=False, fillcolor=bg_color)
        
        # Save
        img_filename = f'khmer_text_{image_idx:05d}.jpg'
        img_path = f'{self.output_dir}/images/{split}/{img_filename}'
        image.save(img_path, quality=random.randint(85, 100))
        
        # Save annotations
        label_filename = f'khmer_text_{image_idx:05d}.txt'
        label_path = f'{self.output_dir}/labels/{split}/{label_filename}'
        with open(label_path, 'w') as f:
            f.write('\n'.join(annotations))
        
        return len(annotations)
    
    def generate_dataset(self, text_file, font_path=None, num_train=800, num_val=200, 
                         min_lines=3, max_lines=10, augment=True):
        """Generate complete dataset"""
        print("Loading text file...")
        texts = self.load_khmer_text(text_file)
        print(f"Loaded {len(texts)} text lines")
        
        if len(texts) < 10:
            print("Warning: Not enough text lines. Add more text to your file.")
        
        print(f"\nSettings: {min_lines}-{max_lines} lines/page, Augment: {augment}, Fonts: {'Random' if not font_path else font_path}")
        
        total_boxes = 0
        
        # Generate Training
        print(f"\nGenerating {num_train} training images...")
        for i in tqdm(range(num_train), desc="Training Set", unit="img"):
            boxes = self.generate_single_image(
                texts, i, 'train', 
                min_lines=min_lines, 
                max_lines=max_lines, 
                augment=augment,
                specific_font_path=font_path
            )
            total_boxes += boxes
        
        # Generate Validation
        print(f"\nGenerating {num_val} validation images...")
        for i in tqdm(range(num_val), desc="Validation Set", unit="img"):
            boxes = self.generate_single_image(
                texts, i, 'val', 
                min_lines=min_lines, 
                max_lines=max_lines, 
                augment=False, 
                specific_font_path=font_path
            )
            total_boxes += boxes
        
        # Create dataset YAML file
        yaml_content = f"""# Khmer Text Detection Dataset
path: {os.path.abspath(self.output_dir)}
train: images/train
val: images/val

# Classes
nc: 1
names: ['text']
"""
        yaml_path = f'{self.output_dir}/data.yaml'
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        
        print(f"\n✓ Dataset generation complete!")
        print(f"✓ Total images: {num_train + num_val}")
        print(f"✓ Total text boxes: {total_boxes}")
        print(f"✓ Dataset config saved to: {yaml_path}")
        print(f"\nNext steps:")
        print(f"1. Review generated images in: {self.output_dir}/images/")
        print(f"2. Train YOLO model with: kiri-ocr train-detector")

def generate_detector_dataset_command(args):
    """
    CLI Command handler for detector dataset generation
    """
    generator = DetectorDatasetGenerator(
        output_dir=args.output,
        fonts_dir=args.fonts_dir
    )
    
    TEXT_FILE = args.text_file
    
    if not os.path.exists(TEXT_FILE):
        print(f"Error: Text file '{TEXT_FILE}' not found!")
        print("Creating sample text file...")
        sample_texts = [
            "ជំរាបសួរ",
            "សូមអរគុណ",
            "អ្នកមានសុខភាពល្អទេ",
            "ខ្ញុំសុខសប្បាយ",
            "តើអ្នកទៅណា",
        ]
        with open(TEXT_FILE, 'w', encoding='utf-8') as f:
            f.write('\n'.join(sample_texts * 20))
        print(f"Created sample file: {TEXT_FILE}")
    
    generator.generate_dataset(
        text_file=TEXT_FILE,
        font_path=args.font,
        num_train=args.num_train,
        num_val=args.num_val,
        min_lines=args.min_lines,
        max_lines=args.max_lines,
        augment=not args.no_augment
    )
