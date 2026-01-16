import os
import sys
import shutil
import random
from collections import Counter
import numpy as np
import cv2
from tqdm import tqdm
from pathlib import Path

# Try to import PIL for better text rendering
try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

class FontManager:
    """Manage font loading from project fonts directory"""
    
    def __init__(self, language='mixed', fonts_dir='fonts'):
        self.language = language
        self.fonts_dir = Path(fonts_dir)
        self.khmer_fonts = []
        self.english_fonts = []
        self.all_fonts = []
        self._load_fonts()
    
    def _load_fonts(self):
        """Load all fonts from fonts directory"""
        
        print(f"\n🔍 Loading fonts from: {self.fonts_dir.absolute()}")
        
        # Create fonts directory if it doesn't exist
        if not self.fonts_dir.exists():
            print(f"  ⚠️  Fonts directory not found: {self.fonts_dir}")
            print(f"  Creating directory...")
            self.fonts_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n  ❌ No fonts found!")
            print(f"\n  Please add .ttf font files to: {self.fonts_dir.absolute()}")
            return
        
        # Get all .ttf and .otf files
        font_files = []
        font_files.extend(self.fonts_dir.glob('*.ttf'))
        font_files.extend(self.fonts_dir.glob('*.TTF'))
        font_files.extend(self.fonts_dir.glob('*.otf'))
        font_files.extend(self.fonts_dir.glob('*.OTF'))
        
        if not font_files:
            print(f"\n  ❌ No font files found in {self.fonts_dir}")
            print(f"\n  Please add .ttf or .otf files to this directory")
            return
        
        print(f"  Found {len(font_files)} font files")
        
        # Categorize fonts by name
        for font_path in font_files:
            font_name = font_path.name.lower()
            
            # Check if it's a Khmer font
            is_khmer = any(keyword in font_name for keyword in [
                'khmer', 'កម្ពុជា', 'battambang', 'siemreap', 
                'bokor', 'moul', 'content', 'metal', 'freehand',
                'fasthand', 'noto', 'kh'
            ])
            
            # Load font at different sizes
            for size in [28, 32, 36, 40, 44, 48]:
                try:
                    font = ImageFont.truetype(str(font_path), size)
                    self.all_fonts.append((str(font_path), size, font))
                    
                    if is_khmer:
                        self.khmer_fonts.append((str(font_path), size, font))
                    else:
                        self.english_fonts.append((str(font_path), size, font))
                    
                except Exception as e:
                    print(f"    ⚠️  Failed to load {font_path.name} at size {size}: {e}")
                    continue
        
        # Print summary
        print(f"\n  📊 Font Summary:")
        print(f"    Total fonts    : {len(self.all_fonts)} (across all sizes)")
        print(f"    Khmer fonts    : {len(self.khmer_fonts)}")
        print(f"    English fonts  : {len(self.english_fonts)}")
    
    def get_random_font(self, text):
        """Get random appropriate font for text"""
        has_khmer = any('\u1780' <= c <= '\u17FF' for c in text)
        
        # Choose from appropriate font pool
        if has_khmer and self.khmer_fonts:
            font_pool = self.khmer_fonts
        elif not has_khmer and self.english_fonts:
            font_pool = self.english_fonts
        else:
            # Fallback to all fonts
            font_pool = self.all_fonts
        
        if not font_pool:
            # Ultimate fallback if no fonts loaded (should error earlier)
            return None
        
        # Return random font
        font_path, size, font = random.choice(font_pool)
        return font

class ImageRenderer:
    """Render text to images"""
    
    def __init__(self, font_manager, image_height=32, image_width=512):
        self.font_manager = font_manager
        self.image_height = image_height
        self.image_width = image_width
    
    def _is_text_supported(self, font, text):
        """Check if font supports the characters in text (detects 'tofu' boxes)"""
        try:
            # Get the glyph for a strictly undefined character to use as reference
            # Using multiple candidates to be safe
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
                # Can't determine reference, assume supported to avoid blocking everything
                return True

            ref_bytes = bytes(ref_mask)

            for char in text:
                # Skip spaces and control characters (they often have empty glyphs which is fine)
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
                    # Error getting mask implies issue
                    return False
                    
            return True
        except Exception:
            # If check fails, be permissive
            return True

    def render(self, text, augment=True, specific_font=None, retry_limit=10):
        """Render text to image using PIL"""
        if not HAS_PIL:
            raise ImportError("Pillow library not found")

        # Get font
        if specific_font:
            font = specific_font
            # Check support for specific font
            if not self._is_text_supported(font, text):
                return None
        else:
            # Retry with retry_limit for random mode
            font = None
            for _ in range(retry_limit):
                candidate = self.font_manager.get_random_font(text)
                if candidate and self._is_text_supported(candidate, text):
                    font = candidate
                    break
            
        if font is None:
             # Just skip
             return None
        
        # Measure text size
        dummy_img = Image.new('L', (1, 1))
        draw = ImageDraw.Draw(dummy_img)
        
        try:
            # New PIL
            bbox = draw.textbbox((0, 0), text, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            offset_y = -bbox[1]
        except AttributeError:
            # Old PIL
            text_w, text_h = draw.textsize(text, font=font)
            offset_y = 0
        
        # Padding
        padding_x = random.randint(10, 30) if augment else 20
        padding_y = random.randint(5, 15) if augment else 10
        
        img_w = text_w + padding_x * 2
        img_h = text_h + padding_y * 2
        
        # Background color
        bg_color = random.randint(235, 255) if augment else 255
        text_color = random.randint(0, 30) if augment else 0
        
        # Create image
        img = Image.new('L', (img_w, img_h), color=bg_color)
        draw = ImageDraw.Draw(img)
        
        # Position
        x = padding_x + (random.randint(-3, 3) if augment else 0)
        y = padding_y + offset_y + (random.randint(-2, 2) if augment else 0)
        
        # Draw text
        draw.text((x, y), text, font=font, fill=text_color)
        
        # Convert to numpy
        img_array = np.array(img)
        
        # Augmentations
        if augment:
            img_array = self._apply_augmentations(img_array, bg_color)
        
        # Resize to target dimensions
        img_array = self._resize_to_target(img_array)
        
        return img_array
    
    def _apply_augmentations(self, img, bg_color):
        """Apply random augmentations"""
        # Gaussian noise
        if random.random() < 0.4:
            noise = np.random.randn(*img.shape) * random.uniform(3, 8)
            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        
        # Gaussian blur
        if random.random() < 0.3:
            kernel_size = random.choice([3, 5])
            img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
        
        # Rotation - Disabled to prevent text cutoff
        # if random.random() < 0.3:
        #     angle = random.uniform(-3, 3)
        #     h, w = img.shape
        #     M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        #     img = cv2.warpAffine(
        #         img, M, (w, h),
        #         borderMode=cv2.BORDER_CONSTANT,
        #         borderValue=int(bg_color)
        #     )
        
        # Morphological operations
        if random.random() < 0.2:
            kernel = np.ones((2, 2), np.uint8)
            if random.random() < 0.5:
                img = cv2.erode(img, kernel, iterations=1)
            else:
                img = cv2.dilate(img, kernel, iterations=1)
        
        # Brightness/Contrast
        if random.random() < 0.3:
            alpha = random.uniform(0.85, 1.15)  # Contrast
            beta = random.randint(-15, 15)       # Brightness
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        
        return img
    
    def _resize_to_target(self, img):
        """Resize to target dimensions"""
        h, w = img.shape[:2]
        
        # Scale to match height
        scale = self.image_height / h
        new_w = int(w * scale)
        
        img = cv2.resize(img, (new_w, self.image_height), interpolation=cv2.INTER_LINEAR)
        
        # Handle width
        if new_w < self.image_width:
            # Pad right
            bg_color = int(np.mean(img[:, -10:])) if new_w > 10 else 255
            padded = np.ones((self.image_height, self.image_width), dtype=np.uint8) * bg_color
            padded[:, :new_w] = img
            img = padded
        elif new_w > self.image_width:
            # Resize to fit
            img = cv2.resize(img, (self.image_width, self.image_height))
        
        return img

class DatasetGenerator:
    """Generate training dataset"""
    
    def __init__(self, language='mixed', image_height=32, image_width=512, fonts_dir='fonts'):
        self.language = language
        self.image_height = image_height
        self.image_width = image_width
        
        self.font_manager = FontManager(language, fonts_dir=fonts_dir)
        self.renderer = ImageRenderer(self.font_manager, image_height, image_width)
    
    def generate_dataset(
        self,
        train_file,
        val_file=None,
        output_dir='data',
        train_augment=100,
        val_augment=1,
        font_mode='random',
        random_augment=False,
        retry_limit=10
    ):
        """Generate complete dataset"""
        
        print("\n" + "="*70)
        print("  📸 Dataset Generation")
        print("="*70)

        # Check for existing dataset and ask user
        append_mode = False
        if os.path.exists(output_dir) and os.listdir(output_dir):
            print(f"\n⚠️  Dataset directory '{output_dir}' already exists.")
            while True:
                try:
                    choice = input("  Do you want to (c)ontinue generating or (s)tart from scratch? [c/s]: ").lower().strip()
                    if choice == 'c':
                        append_mode = True
                        print("  🔄 Continuing generation (appending to existing)...")
                        break
                    elif choice == 's':
                        print("  🗑️  Cleaning up existing directory...")
                        shutil.rmtree(output_dir)
                        break
                except KeyboardInterrupt:
                    print("\nAborted.")
                    return
        
        # Create output dirs
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/train/images", exist_ok=True)
        os.makedirs(f"{output_dir}/val/images", exist_ok=True)
        
        # Generate training set
        print("\n📚 Generating TRAINING set...")
        train_data = self._generate_split(
            train_file,
            f"{output_dir}/train",
            augment_factor=train_augment,
            split_name='train',
            font_mode=font_mode,
            random_augment=random_augment,
            retry_limit=retry_limit,
            append=append_mode
        )
        
        # Generate validation set
        if val_file and os.path.exists(val_file):
            print("\n📚 Generating VALIDATION set (from separate file)...")
            val_data = self._generate_split(
                val_file,
                f"{output_dir}/val",
                augment_factor=val_augment,
                split_name='val',
                retry_limit=retry_limit,
                append=append_mode
            )
        else:
            # Split from training
            print("\n📚 Generating VALIDATION set (split from training)...")
            with open(train_file, 'r', encoding='utf-8') as f:
                all_lines = [line.strip() for line in f if line.strip()]
            
            random.shuffle(all_lines)
            split_idx = int(len(all_lines) * 0.9)
            val_lines = all_lines[split_idx:]
            
            # Save temporary val file
            temp_val = f"{output_dir}/temp_val.txt"
            with open(temp_val, 'w', encoding='utf-8') as f:
                f.write('\n'.join(val_lines))
            
            val_data = self._generate_split(
                temp_val,
                f"{output_dir}/val",
                augment_factor=val_augment,
                split_name='val',
                retry_limit=retry_limit,
                append=append_mode
            )
            
            os.remove(temp_val)
        
        print("\n" + "="*70)
        print("  ✅ Dataset Generation Complete!")
        print("="*70)
        print(f"  Train: {len(train_data):,} samples")
        print(f"  Val:   {len(val_data):,} samples")
        print(f"  Output: {output_dir}")
        print("="*70 + "\n")
    
    def _generate_split(self, text_file, output_dir, augment_factor, split_name, font_mode='random', random_augment=False, retry_limit=10, append=False):
        """Generate one split (train/val)"""
        
        # Read text lines
        with open(text_file, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        print(f"  Loaded {len(lines)} lines from {text_file}")
        
        # Load existing progress if appending
        existing_counts = Counter()
        start_idx = 0
        
        if append and os.path.exists(f"{output_dir}/images"):
            # 1. Determine start index
            existing_files = [f for f in os.listdir(f"{output_dir}/images") if f.endswith('.png')]
            if existing_files:
                indices = []
                prefix = f"{split_name}_"
                for fname in existing_files:
                    if fname.startswith(prefix):
                        try:
                            num_part = fname[len(prefix):-4]
                            indices.append(int(num_part))
                        except ValueError:
                            continue
                
                if indices:
                    start_idx = max(indices) + 1
                    print(f"  👉 Appending starting from index {start_idx}")
            
            # 2. Count existing samples per text line
            labels_path = f"{output_dir}/labels.txt"
            if os.path.exists(labels_path):
                print(f"  📖 Analyzing existing labels to resume...")
                try:
                    with open(labels_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            parts = line.strip().split('\t', 1)
                            if len(parts) == 2:
                                existing_counts[parts[1]] += 1
                    print(f"  ✓ Found {sum(existing_counts.values())} existing samples")
                except Exception as e:
                    print(f"  ⚠️ Could not read labels file: {e}")

        # Create samples
        samples = []
        skipped_count = 0
        
        if font_mode == 'all':
            print(f"  Using ALL fonts mode (augment factor ignored for font selection)")
            fonts_list = self.font_manager.all_fonts
            print(f"  Iterating {len(fonts_list)} fonts per line...")
            
            for line in lines:
                # In 'all' mode, we skip line only if completely done to avoid missing fonts
                # (Since we don't track which fonts were done)
                if append and existing_counts[line] >= len(fonts_list):
                    skipped_count += len(fonts_list)
                    continue
                    
                for font_tuple in fonts_list:
                    samples.append({'text': line, 'font': font_tuple[2]})
        else:
            # Random mode
            print(f"  Using RANDOM fonts mode (retry limit: {retry_limit} attempts per sample)")
            for line in lines:
                needed = augment_factor
                have = existing_counts[line] if append else 0
                
                remaining = max(0, needed - have)
                
                if remaining < needed:
                    skipped_count += (needed - remaining)
                
                for _ in range(remaining):
                    samples.append({'text': line, 'font': None})
        
        if skipped_count > 0:
            print(f"  ⏭️  Skipping {skipped_count} already generated samples")
            
        random.shuffle(samples)

        # Generate images
        file_mode = 'a' if append else 'w'
        labels_file = open(f"{output_dir}/labels.txt", file_mode, encoding='utf-8')
        success_count = 0
        
        print(f"  Generating {len(samples)} images...")
        
        for idx, sample in enumerate(tqdm(samples, desc="    Generating", unit="img")):
            text = sample['text']
            specific_font = sample['font']
            
            # Set retry_limit based on font_mode
            current_retry_limit = 1 if font_mode == 'all' else retry_limit
            
            try:
                # Render image
                img = self.renderer.render(
                    text,
                    augment=((augment_factor > 1) or random_augment),
                    specific_font=specific_font,
                    retry_limit=current_retry_limit
                )
                
                if img is None:
                    continue
                
                # Save
                current_idx = start_idx + idx
                img_filename = f"{split_name}_{current_idx:06d}.png"
                img_path = f"{output_dir}/images/{img_filename}"
                cv2.imwrite(img_path, img)
                
                # Write label
                labels_file.write(f"{img_filename}\t{text}\n")
                success_count += 1
                
            except Exception as e:
                print(f"    ⚠ Failed for '{text[:30]}...': {e}")
                continue
        
        labels_file.close()
        print(f"  ✓ Generated {success_count} / {len(samples)} images\n")
        
        return samples[:success_count]

def generate_command(args):
    # Check files exist
    if not os.path.exists(args.train_file):
        print(f"\n❌ Error: Training file not found: {args.train_file}\n")
        return 1
    
    if not HAS_PIL:
        print("❌ PIL/Pillow not found - install with: pip install Pillow")
        return 1
    
    # Get retry limit from args, default to 10
    retry_limit = getattr(args, 'retry_limit', 10)
        
    # Generate
    generator = DatasetGenerator(
        language=args.language,
        image_height=args.height,
        image_width=args.width,
        fonts_dir=args.fonts_dir
    )
    
    generator.generate_dataset(
        train_file=args.train_file,
        val_file=args.val_file,
        output_dir=args.output,
        train_augment=args.augment,
        val_augment=args.val_augment,
        font_mode=args.font_mode if hasattr(args, 'font_mode') else 'random',
        random_augment=args.random_augment if hasattr(args, 'random_augment') else False,
        retry_limit=retry_limit
    )
    return 0