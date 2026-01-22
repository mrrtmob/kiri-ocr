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
    
    def load_text(self, text_file):
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
        """Check if font supports the characters in text"""
        try:
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
                return True

            ref_bytes = bytes(ref_mask)

            for char in text:
                if char.isspace() or ord(char) < 32:
                    continue
                    
                try:
                    char_mask = font.getmask(char)
                    char_bbox = char_mask.getbbox()
                    
                    if char_bbox == ref_bbox:
                        if bytes(char_mask) == ref_bytes:
                            return False
                except Exception:
                    return False
                    
            return True
        except Exception:
            return True
    
    def apply_augmentation(self, image):
        """Apply random augmentations to the full page image"""
        img_array = np.array(image)
        
        # Gaussian Noise
        if random.random() > 0.5:
            noise = np.random.normal(0, random.randint(5, 20), img_array.shape)
            img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
            
        # Blur
        if random.random() > 0.6:
            ksize = random.choice([3, 5])
            img_array = cv2.GaussianBlur(img_array, (ksize, ksize), 0)
            
        # Brightness/Contrast
        if random.random() > 0.5:
            alpha = random.uniform(0.8, 1.2)
            beta = random.randint(-30, 30)
            img_array = cv2.convertScaleAbs(img_array, alpha=alpha, beta=beta)
            
        # JPEG Compression
        if random.random() > 0.5:
            quality = random.randint(50, 95)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            result, encimg = cv2.imencode('.jpg', img_array, encode_param)
            img_array = cv2.imdecode(encimg, 1)

        return Image.fromarray(img_array)
    
    def measure_line_bbox(self, draw, line_parts, font, start_x, start_y):
        """Measure the complete bounding box for a line with multiple text parts"""
        current_x = start_x
        min_x = start_x
        max_x = start_x
        min_y = start_y
        max_y = start_y
        
        for text_part in line_parts:
            try:
                try:
                    bbox = draw.textbbox((current_x, start_y), text_part, font=font)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                    
                    min_x = min(min_x, bbox[0])
                    max_x = max(max_x, bbox[2])
                    min_y = min(min_y, bbox[1])
                    max_y = max(max_y, bbox[3])
                    
                except AttributeError:
                    text_width, text_height = draw.textsize(text_part, font=font)
                    max_x = current_x + text_width
                    max_y = start_y + text_height
                
                current_x += text_width + random.randint(10, 30)
                
            except Exception:
                continue
        
        total_width = max_x - min_x
        total_height = max_y - min_y
        
        return min_x, min_y, max_x, max_y, total_width, total_height
    
    def generate_single_image(self, texts, image_idx, split='train', 
                              min_lines=5, max_lines=15, augment=True, 
                              specific_font_path=None,
                              min_words_per_line=2, max_words_per_line=5):
        """Generate realistic document-style layout with paragraphs and tight line spacing"""
        
        # Image dimensions (more document-like)
        img_width = random.randint(800, 1200)
        img_height = random.randint(800, 1400)
        
        # Create image
        bg_color = self.get_random_background_color()
        image = Image.new('RGB', (img_width, img_height), bg_color)
        draw = ImageDraw.Draw(image)
        
        # Determine number of lines
        num_lines = random.randint(min_lines, max_lines)
        
        # Annotations
        annotations = []
        
        # Document-style layout
        y_offset = random.randint(60, 120)
        x_margin = random.randint(60, 120)
        
        # REALISTIC DOCUMENT SPACING
        # Within paragraph: tight spacing (like line-height: 1.2-1.5)
        # Between paragraphs: larger spacing
        
        # Common text color and font for document consistency
        text_color = self.get_random_text_color()
        
        # Font paths
        available_font_paths = []
        if self.font_manager and self.font_manager.all_fonts:
            available_font_paths = list(set([f[0] for f in self.font_manager.all_fonts]))
        
        # Select ONE font size for the entire document (realistic)
        document_font_size = random.randint(32, 48)
        
        successful_lines = 0
        current_paragraph_lines = 0
        paragraph_size = random.randint(2, 5)  # Lines per paragraph
        
        while successful_lines < num_lines and y_offset < img_height - 100:
            # Create line with multiple words
            num_words = random.randint(min_words_per_line, max_words_per_line)
            line_texts = random.sample(texts, min(num_words, len(texts)))
            
            # Use document font size
            font_size = document_font_size
            font = None
            
            # Try specific font
            if specific_font_path:
                try:
                    candidate = ImageFont.truetype(specific_font_path, font_size)
                    all_supported = all(self._is_text_supported(candidate, t) for t in line_texts)
                    if all_supported:
                        font = candidate
                except:
                    pass
            
            # Random font selection
            if font is None and available_font_paths:
                retries = 10
                for _ in range(retries):
                    random_path = random.choice(available_font_paths)
                    try:
                        candidate = ImageFont.truetype(random_path, font_size)
                        all_supported = all(self._is_text_supported(candidate, t) for t in line_texts)
                        if all_supported:
                            font = candidate
                            break
                    except:
                        continue
            
            if font is None:
                font = ImageFont.load_default()
            
            # Document-style alignment (mostly left-aligned)
            align_choice = random.choices(['left', 'justified', 'center'], weights=[70, 25, 5])[0]
            
            # Pre-measure the line
            temp_x = x_margin
            min_x, min_y, max_x, max_y, total_width, total_height = \
                self.measure_line_bbox(draw, line_texts, font, temp_x, y_offset)
            
            # Skip if too small or too wide
            if total_height < 20 or total_width > img_width - 2 * x_margin:
                continue
            
            # Calculate X position
            if align_choice == 'left' or align_choice == 'justified':
                x_offset = x_margin + random.randint(0, 10)
            elif align_choice == 'center':
                x_offset = (img_width - total_width) // 2
            else:
                max_x_start = img_width - total_width - x_margin
                x_offset = random.randint(x_margin, max(x_margin + 1, max_x_start))
            
            x_offset = max(10, min(x_offset, img_width - total_width - 10))
            
            # Check vertical space
            if y_offset + total_height > img_height - 60:
                break
            
            # Re-measure with actual position
            min_x, min_y, max_x, max_y, total_width, total_height = \
                self.measure_line_bbox(draw, line_texts, font, x_offset, y_offset)
            
            # Draw all words in the line
            current_x = x_offset
            word_spacing = random.randint(12, 25)
            
            for text_part in line_texts:
                try:
                    draw.text((current_x, y_offset), text_part, font=font, fill=text_color)
                    
                    try:
                        bbox = draw.textbbox((current_x, y_offset), text_part, font=font)
                        text_width = bbox[2] - bbox[0]
                    except AttributeError:
                        text_width, _ = draw.textsize(text_part, font=font)
                    
                    current_x += text_width + word_spacing
                    
                except Exception:
                    continue
            
            # Create bounding box with minimal padding (document-style)
            BBOX_PADDING = 4  # Tighter padding for document look
            x1 = max(0, min_x - BBOX_PADDING)
            y1 = max(0, min_y - BBOX_PADDING)
            x2 = min(img_width, max_x + BBOX_PADDING)
            y2 = min(img_height, max_y + BBOX_PADDING)
            
            # YOLO format
            x_center = ((x1 + x2) / 2) / img_width
            y_center = ((y1 + y2) / 2) / img_height
            width = (x2 - x1) / img_width
            height = (y2 - y1) / img_height
            
            if 0 < x_center < 1 and 0 < y_center < 1 and 0 < width < 1 and 0 < height < 1:
                annotations.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
                successful_lines += 1
                current_paragraph_lines += 1
            
            # REALISTIC DOCUMENT LINE SPACING
            # Calculate line-height based on text height
            base_line_height = int(total_height * random.uniform(1.2, 1.5))  # Standard line-height
            
            # Check if we should start a new paragraph
            if current_paragraph_lines >= paragraph_size:
                # Paragraph break: add extra spacing
                paragraph_spacing = random.randint(8, 16)
                y_offset = y2 + paragraph_spacing
                
                # Reset paragraph tracking
                current_paragraph_lines = 0
                paragraph_size = random.randint(2, 5)
            else:
                # Within paragraph: tight spacing (realistic document)
                # This creates the effect where 2 lines might be detected as 1
                within_para_spacing = random.randint(2, 8)  # Very tight! 
                y_offset = y2 + within_para_spacing
        
        # Apply augmentation
        if augment:
            image = self.apply_augmentation(image)
        
        # Minimal rotation for document scan effect
        if augment and random.random() > 0.85:
            angle = random.uniform(-0.8, 0.8)  # Very slight rotation
            image = image.rotate(angle, expand=False, fillcolor=bg_color)
        
        # Save
        img_filename = f'image_{image_idx:05d}.jpg'
        img_path = f'{self.output_dir}/images/{split}/{img_filename}'
        image.save(img_path, quality=random.randint(85, 100))
        
        label_filename = f'image_{image_idx:05d}.txt'
        label_path = f'{self.output_dir}/labels/{split}/{label_filename}'
        with open(label_path, 'w') as f:
            f.write('\n'.join(annotations))
        
        return len(annotations)
    
    def generate_dataset(self, text_file, font_path=None, num_train=800, num_val=200, 
                         min_lines=5, max_lines=15, min_words_per_line=2, 
                         max_words_per_line=5, augment=True):
        """Generate complete dataset with realistic document formatting"""
        print("Loading text file...")
        texts = self.load_text(text_file)
        print(f"Loaded {len(texts)} text segments")
        
        if len(texts) < max_words_per_line:
            print(f"Warning: Not enough text segments. Need at least {max_words_per_line}")
        
        print(f"\n=== Document-Style Dataset Generation ===")
        print(f"Settings:")
        print(f"  • Lines per image: {min_lines}-{max_lines}")
        print(f"  • Words per line: {min_words_per_line}-{max_words_per_line}")
        print(f"  • Layout: Realistic document with paragraphs")
        print(f"  • Within-paragraph spacing: 2-8px (tight)")
        print(f"  • Between-paragraph spacing: 20-40px")
        print(f"  • Augmentation: {augment}")
        
        total_boxes = 0
        
        # Generate Training
        print(f"\nGenerating {num_train} training images...")
        for i in tqdm(range(num_train), desc="Training Set", unit="img"):
            boxes = self.generate_single_image(
                texts, i, 'train', 
                min_lines=min_lines, 
                max_lines=max_lines, 
                augment=augment,
                specific_font_path=font_path,
                min_words_per_line=min_words_per_line,
                max_words_per_line=max_words_per_line
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
                specific_font_path=font_path,
                min_words_per_line=min_words_per_line,
                max_words_per_line=max_words_per_line
            )
            total_boxes += boxes
        
        # Create YAML
        yaml_content = f"""# Khmer Text Line Detection Dataset (Document-Style)
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
        print(f"✓ Total line boxes: {total_boxes}")
        print(f"✓ Avg lines per image: {total_boxes/(num_train+num_val):.1f}")
        print(f"✓ Config: {yaml_path}")
        print(f"\n=== Document Formatting Features ===")
        print(f"✓ Paragraph structure (2-5 lines per paragraph)")
        print(f"✓ Tight within-paragraph spacing (2-8px)")
        print(f"✓ Realistic line-height (1.2-1.5x text height)")
        print(f"✓ Paragraph breaks with extra spacing (20-40px)")
        print(f"✓ Consistent font size per document")
        print(f"✓ Left-aligned text (70%), justified (25%), centered (5%)")
        print(f"✓ {min_words_per_line}-{max_words_per_line} words per line")
        print(f"\nThis mimics real documents where closely-spaced lines")
        print(f"might be detected as a single text block!")
        print(f"\nNext: Train with 'kiri-ocr train-detector'")

def generate_detector_dataset_command(args):
    """CLI Command handler"""
    generator = DetectorDatasetGenerator(
        output_dir=args.output,
        fonts_dir=args.fonts_dir
    )
    
    TEXT_FILE = args.text_file
    
    if not os.path.exists(TEXT_FILE):
        print(f"Error: Text file '{TEXT_FILE}' not found!")
        print("Creating sample text file...")
        sample_texts = [
            "ជំរាបសួរ", "សូមអរគុណ", "អ្នកមានសុខភាពល្អទេ",
            "ខ្ញុំសុខសប្បាយ", "តើអ្នកទៅណា", "យើងជួបគ្នាម្តងទៀត",
            "សូមអត់ទោស", "ខ្ញុំមិនយល់ទេ", "តើអ្នកនិយាយភាសាអង់គ្លេសបានទេ",
            "សូមជួយខ្ញុំផង", "ធ្វើដំណើរប្រុងប្រយ័ត្ន", "ជាទីស្រលាញ់",
            "នេះគឺជាឯកសារ", "របស់យើង", "សរសេរជាភាសាខ្មែរ",
            "ក្នុងឆ្នាំ២០២៦", "ខែមករា", "ថ្ងៃទី២២",
        ]
        with open(TEXT_FILE, 'w', encoding='utf-8') as f:
            f.write('\n'.join(sample_texts * 10))
        print(f"Created sample file: {TEXT_FILE}")
    
    min_words = getattr(args, 'min_words_per_line', 2)
    max_words = getattr(args, 'max_words_per_line', 5)
    
    generator.generate_dataset(
        text_file=TEXT_FILE,
        font_path=args.font,
        num_train=args.num_train,
        num_val=args.num_val,
        min_lines=args.min_lines,
        max_lines=args.max_lines,
        min_words_per_line=min_words,
        max_words_per_line=max_words,
        augment=not args.no_augment
    )