import torch
import cv2
import numpy as np
import sys
from PIL import Image
from pathlib import Path
import json

from .model import LightweightOCR, CharacterSet

class OCR:
    """Complete Document OCR System with Padding"""
    
    # Cache loaded models to avoid reloading in the same process
    _model_cache = {}

    def __init__(self, model_path='mrrtmob/kiri-ocr',
                 charset_path='models/charset_lite.txt',
                 language='mixed',
                 padding=10,
                 device='cpu',
                 verbose=False):
        """
        Args:
            model_path: Path to trained model (.kiri or .pth)
            charset_path: Path to character set (used if model doesn't contain charset)
            language: 'english', 'khmer', or 'mixed'
            padding: Pixels to pad around detected boxes (default: 10)
            device: 'cpu' or 'cuda'
            verbose: Whether to print loading/processing info
        """
        self.device = device
        self.verbose = verbose
        self.language = language
        self.padding = padding
        
        # Resolve model path
        model_file = Path(model_path)
        if not model_file.exists():
             # Try looking in package directory
             pkg_dir = Path(__file__).parent
             if (pkg_dir / model_path).exists():
                 model_path = str(pkg_dir / model_path)
             # Try looking in sibling 'models' package (if installed via setup.py with models package)
             elif (pkg_dir.parent / 'models' / model_file.name).exists():
                 model_path = str(pkg_dir.parent / 'models' / model_file.name)
             # Fallback to .pth if .kiri not found
             elif model_path.endswith('.kiri') and (model_file.parent / 'model.pth').exists():
                 model_path = str(model_file.parent / 'model.pth')
             # Check if it's a Hugging Face repo ID (e.g. "username/repo")
             elif "/" in model_path and not model_path.startswith(".") and not model_path.startswith("/"):
                 try:
                     from huggingface_hub import try_to_load_from_cache
                     # Quick check without downloading
                     cached = try_to_load_from_cache(model_path, "model.kiri")
                     if cached and str(cached) != "_CACHED_NO_EXIST":
                         model_path = cached
                 except ImportError:
                     pass  # Fall back to current method

                 try:
                     from huggingface_hub import hf_hub_download
                     
                     # Try local cache first
                     try:
                        resolved_path = hf_hub_download(repo_id=model_path, filename="model.kiri", local_files_only=True)
                        if self.verbose:
                             print(f"📦 Found model in local cache: {model_path}")
                        model_path = resolved_path
                     except Exception:
                        # Fallback to online
                        if self.verbose:
                            print(f"⬇️ Downloading model from Hugging Face: {model_path}")
                        
                        # Download config.json to ensure download stats are tracked on HF
                        try:
                            hf_hub_download(repo_id=model_path, filename="config.json")
                        except Exception:
                            # Config might not exist for older uploads, ignore
                            pass
                            
                        model_path = hf_hub_download(repo_id=model_path, filename="model.kiri")
                 except Exception as e:
                     if self.verbose:
                         print(f"⚠️ Could not download from Hugging Face: {e}")
        
        # Check memory cache
        cache_key = (str(model_path), device)
        if cache_key in OCR._model_cache:
            if self.verbose:
                print(f"⚡ Loading model from memory cache")
            self.model, self.charset = OCR._model_cache[cache_key]
        else:
            if self.verbose:
                print(f"📦 Loading OCR model from {model_path}...")
            
            try:
                checkpoint = torch.load(model_path, map_location=device, weights_only=False)
                
                # Load charset
                if 'charset' in checkpoint:
                    if self.verbose:
                        print(f"  ✓ Found embedded charset ({len(checkpoint['charset'])} chars)")
                    self.charset = CharacterSet.from_checkpoint(checkpoint)
                else:
                    # Fallback to charset file
                    if not Path(charset_path).exists():
                        # Try looking in package directory if not found locally
                        pkg_dir = Path(__file__).parent
                        if (pkg_dir / charset_path).exists():
                            charset_path = str(pkg_dir / charset_path)
                    
                    if self.verbose:
                        print(f"  ℹ️ Loading charset from file: {charset_path}")
                    self.charset = CharacterSet.load(charset_path)
                
                # Initialize model
                self.model = LightweightOCR(num_chars=len(self.charset)).eval()
                
                # Load weights
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    # Assume checkpoint IS state_dict (legacy/raw save)
                    self.model.load_state_dict(checkpoint)
                
                self.model = self.model.to(device)
                
                # Update cache
                OCR._model_cache[cache_key] = (self.model, self.charset)
                
                if self.verbose:
                    print(f"✓ Model loaded ({len(self.charset)} characters)")
                    print(f"✓ Box padding: {padding}px")

            except RuntimeError as e:
                if "size mismatch" in str(e):
                    print(f"\n❌ Error loading model: {e}")
                    print("\n⚠️  CRITICAL: The model weights do not match the character set.")
                    print(f"    - Charset size: {len(self.charset)}")
                    print(f"    - Model file:   {model_path}")
                    sys.exit(1)
                else:
                    raise e
            except Exception as e:
                print(f"❌ Error loading model: {e}")
                raise e
        
        # Detector
        self._detector = None
    
    @property
    def detector(self):
        """Lazy-load detector only when needed"""
        if self._detector is None:
            from .detector import TextDetector
            self._detector = TextDetector()
        return self._detector

    def _preprocess_region(self, img, box, extra_padding=5):
        """
        Crop and preprocess a region with extra padding
        
        Args:
            img: Source image (numpy array)
            box: Bounding box (x, y, w, h) - already has padding from detector
            extra_padding: Additional padding when cropping (default: 5px)
        """
        img_h, img_w = img.shape[:2]
        x, y, w, h = box
        
        # Add extra padding (with boundary checks)
        x_extra = max(0, x - extra_padding)
        y_extra = max(0, y - extra_padding)
        w_extra = min(img_w - x_extra, w + 2 * extra_padding)
        h_extra = min(img_h - y_extra, h + 2 * extra_padding)
        
        # Crop
        roi = img[y_extra:y_extra+h_extra, x_extra:x_extra+w_extra]
        
        if roi.size == 0:
            return None
        
        # Invert if dark background (Model expects Light BG / Dark Text)
        if np.mean(roi) < 127:
             roi = 255 - roi

        # Convert to PIL
        roi_pil = Image.fromarray(roi).convert('L')
        
        # Resize maintaining aspect ratio
        orig_w, orig_h = roi_pil.size
        new_h = 32
        new_w = int((orig_w / orig_h) * new_h)
        
        # Ensure minimum width
        if new_w < 32:
            new_w = 32
        
        roi_pil = roi_pil.resize((new_w, new_h), Image.LANCZOS)
        
        # Normalize
        roi_array = np.array(roi_pil) / 255.0
        roi_tensor = torch.tensor(roi_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        return roi_tensor
    
    def recognize_single_line_image(self, image_path):
        """Recognize text from a single line image without detection"""
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
            
        # Preprocess
        # We need to resize height to 32 and keep aspect ratio
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
        # Invert if dark background (Model expects Light BG / Dark Text)
        if np.mean(img) < 127:
             img = 255 - img

        img_pil = Image.fromarray(img)
        w, h = img_pil.size
        new_h = 32
        new_w = int((w / h) * new_h)
        if new_w < 32: new_w = 32
        
        img_pil = img_pil.resize((new_w, new_h), Image.LANCZOS)
        img_array = np.array(img_pil) / 255.0
        img_tensor = torch.tensor(img_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        text, confidence = self.recognize_region(img_tensor)
        return text, confidence

    def recognize_region(self, image_tensor):
        """Recognize text in a single region"""
        image_tensor = image_tensor.to(self.device)
        
        with torch.no_grad():
            logits = self.model(image_tensor)
            
            # CTC decoding (LightweightOCR)
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
            preds = preds.squeeze().tolist()
            
            # Handle single timestep
            if not isinstance(preds, list):
                preds = [preds]
            
            # CTC decode
            char_confidences = []
            decoded_indices = []
            previous_idx = -1
            
            for i, idx in enumerate(preds):
                if idx != previous_idx:
                    if idx > 2:  # Skip BLANK, PAD, SOS
                        decoded_indices.append(idx)
                        char_confidences.append(probs[i, 0, idx].item())
                previous_idx = idx
            
            confidence = np.mean(char_confidences) if char_confidences else 0.0
            text = self.charset.decode(decoded_indices)
        
        return text, confidence
    
    def process_document(self, image_path, mode='lines', verbose=False):
        """
        Process entire document
        
        Args:
            image_path: Path to document image
            mode: 'lines' or 'words' for detection granularity
            verbose: Whether to print progress
        
        Returns:
            List of dicts with 'box', 'text', 'confidence'
        """
        if verbose:
            print(f"\n📄 Processing document: {image_path}")
            print(f"🔲 Box padding: {self.padding}px")
        
        # Detect text regions
        if mode == 'lines':
            boxes = self.detector.detect_lines(image_path)
        else:
            boxes = self.detector.detect_words(image_path)
        
        if verbose:
            print(f"🔍 Detected {len(boxes)} regions")
        
        # Load image
        img = cv2.imread(str(image_path))
        
        # Recognize each region
        results = []
        for i, box in enumerate(boxes, 1):
            try:
                # Preprocess with extra padding
                region_tensor = self._preprocess_region(img, box, extra_padding=5)
                if region_tensor is None:
                    continue
                
                # Recognize
                text, confidence = self.recognize_region(region_tensor)
                
                # Convert types for JSON serialization
                safe_box = [int(v) for v in box]
                safe_confidence = float(confidence)
                
                results.append({
                    'box': safe_box,
                    'text': text,
                    'confidence': safe_confidence,
                    'line_number': i
                })
                
                if verbose:
                    print(f"  {i:2d}. {text:50s} ({confidence*100:.1f}%)")
                
            except Exception as e:
                if verbose:
                    print(f"  {i:2d}. [Error: {e}]")
                continue
        
        return results
    
    def extract_text(self, image_path, mode='lines', verbose=False):
        """Extract all text from document as string"""
        results = self.process_document(image_path, mode, verbose=verbose)
        
        if not results:
            return "", results
            
        # Reconstruct text layout
        # Sort by Y then X
        results.sort(key=lambda r: (r['box'][1], r['box'][0]))
        
        full_text = ""
        for i, res in enumerate(results):
            text = res['text']
            box = res['box']
            _, y, _, h = box
            
            if i > 0:
                prev_box = results[i-1]['box']
                prev_y, prev_h = prev_box[1], prev_box[3]
                
                # Check if on the same line (vertical center is close)
                center_y = y + h/2
                prev_center_y = prev_y + prev_h/2
                
                # If vertical centers are close (within half height), assume same line
                if abs(center_y - prev_center_y) < max(h, prev_h) / 2:
                    full_text += " " + text
                else:
                    full_text += "\n" + text
            else:
                full_text += text
        
        return full_text, results
