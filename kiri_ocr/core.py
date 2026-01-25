# kiri_ocr/core.py
"""
Core OCR class for Kiri OCR using Transformer architecture.
"""
import torch
import cv2
import numpy as np
import sys
from PIL import Image
from pathlib import Path
import json

try:
    from safetensors.torch import load_file
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False

from .model import (
    KiriOCR,
    CFG,
    CharTokenizer,
    beam_decode_one_batched,
    greedy_ctc_decode,
    preprocess_pil,
)


class OCR:
    """Complete Document OCR System with Transformer Model"""

    _model_cache = {}

    def __init__(
        self,
        model_path="mrrtmob/kiri-ocr",
        det_model_path=None,
        det_method="db",
        det_conf_threshold=0.5,
        padding=10,
        device="cpu",
        verbose=False,
        use_beam_search=True,
        use_fp16=None,
    ):
        """
        Args:
            model_path: Path to trained model (.safetensors or .pt) or HuggingFace repo
            det_model_path: Path to detector model (optional)
            det_method: Detection method ('db', 'craft', 'yolo')
            det_conf_threshold: Confidence threshold for detector
            padding: Pixels to pad around detected boxes
            device: 'cpu' or 'cuda'
            verbose: Print loading info
            use_beam_search: Use beam search (True) or greedy CTC (False)
            use_fp16: Force FP16 usage (True/False/None)
        """
        self.device = device
        self.verbose = verbose
        self.padding = padding
        self.det_model_path = det_model_path
        self.det_method = det_method
        self.det_conf_threshold = det_conf_threshold
        self.use_beam_search = use_beam_search
        self.use_fp16 = use_fp16

        # Transformer model components
        self.cfg = None
        self.tokenizer = None
        self.model = None

        # Store repo_id for detector lazy loading
        self.repo_id = None
        if "/" in model_path and not model_path.startswith((".", "/")):
            self.repo_id = model_path

        # Resolve model path (HuggingFace, local, etc.)
        model_path = self._resolve_model_path(model_path)

        # Load model
        self._load_model(model_path)

        # Lazy-loaded detector
        self._detector = None

    def _resolve_model_path(self, model_path):
        """Resolve model path from various sources"""
        model_file = Path(model_path)

        if model_file.exists():
            return str(model_file)

        # Try package directory
        pkg_dir = Path(__file__).parent
        candidates = [
            pkg_dir / model_path,
            pkg_dir.parent / "models" / model_file.name,
        ]

        for candidate in candidates:
            if candidate.exists():
                return str(candidate)

        # Try HuggingFace
        if "/" in model_path and not model_path.startswith((".", "/")):
            try:
                from huggingface_hub import hf_hub_download

                if self.verbose:
                    print(f"⬇️ Downloading from HuggingFace: {model_path}")

                # Try config.json for download stats
                try:
                    hf_hub_download(repo_id=model_path, filename="config.json")
                except:
                    pass

                # Try downloading vocab.json or vocab_auto.json
                for vocab_name in ["vocab.json", "vocab_auto.json"]:
                    try:
                        hf_hub_download(repo_id=model_path, filename=vocab_name)
                        break
                    except:
                        pass

                # Try downloading model (safetensors first, then pt)
                for model_name in ["model.safetensors", "model.pt"]:
                    try:
                        return hf_hub_download(repo_id=model_path, filename=model_name)
                    except:
                        pass

            except Exception as e:
                if self.verbose:
                    print(f"⚠️ HuggingFace download failed: {e}")

        return model_path  # Return as-is, let load fail with clear error

    def _load_model(self, model_path):
        """Load Transformer model from checkpoint (safetensors or pt)"""
        cache_key = (str(model_path), self.device)

        if cache_key in OCR._model_cache:
            if self.verbose:
                print(f"⚡ Loading from memory cache")
            cached = OCR._model_cache[cache_key]
            self.model = cached["model"]
            self.cfg = cached["cfg"]
            self.tokenizer = cached["tokenizer"]
            return

        if self.verbose:
            print(f"📦 Loading OCR model from {model_path}...")

        try:
            is_safetensors = model_path.endswith('.safetensors')
            
            if is_safetensors and HAS_SAFETENSORS:
                # Load safetensors format
                state_dict = load_file(model_path, device=self.device)
                
                # Load metadata from JSON file
                metadata_path = model_path.replace('.safetensors', '_meta.json')
                self.cfg = CFG()
                vocab_path = ""
                
                if Path(metadata_path).exists():
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    vocab_path = metadata.get("vocab_path", "")
                    config_data = metadata.get("config", {})
                    if config_data:
                        self.cfg.IMG_H = config_data.get("IMG_H", self.cfg.IMG_H)
                        self.cfg.IMG_W = config_data.get("IMG_W", self.cfg.IMG_W)
                        self.cfg.USE_CTC = config_data.get("USE_CTC", self.cfg.USE_CTC)
                        self.cfg.USE_FP16 = config_data.get("USE_FP16", self.cfg.USE_FP16)
            else:
                # Load torch checkpoint
                checkpoint = torch.load(
                    model_path, map_location=self.device, weights_only=False
                )

                # Extract config and state dict
                if "config" in checkpoint:
                    self.cfg = checkpoint["config"]
                    state_dict = checkpoint["model"]
                    vocab_path = checkpoint.get("vocab_path", "")
                else:
                    self.cfg = CFG()
                    state_dict = checkpoint
                    vocab_path = ""

            # Override FP16 setting if requested
            if self.use_fp16 is not None:
                self.cfg.USE_FP16 = self.use_fp16

            # Find vocab file
            vocab_path = self._find_vocab_file(vocab_path, model_path)

            if not vocab_path or not Path(vocab_path).exists():
                raise FileNotFoundError(
                    f"Could not find vocabulary file for model. "
                    f"Expected near: {model_path}"
                )

            self.tokenizer = CharTokenizer(vocab_path, self.cfg)
            self.model = KiriOCR(self.cfg, self.tokenizer).to(self.device)
            self.model.load_state_dict(state_dict, strict=True)
            self.model.eval()

            if self.cfg.USE_FP16 and self.device == "cuda":
                self.model.half()

            if self.verbose:
                print(f"  ✓ Loaded (Vocab: {self.tokenizer.vocab_size} chars)")

            # Cache
            OCR._model_cache[cache_key] = {
                "model": self.model,
                "cfg": self.cfg,
                "tokenizer": self.tokenizer,
            }

        except RuntimeError as e:
            if "size mismatch" in str(e):
                print(f"\n❌ Model/vocab size mismatch: {e}")
                sys.exit(1)
            raise

    def _find_vocab_file(self, vocab_path, model_path):
        """Find vocabulary file for model"""
        model_dir = Path(model_path).parent

        candidates = [
            vocab_path,
            model_dir / Path(vocab_path).name if vocab_path else None,
            model_dir / "vocab.json",
            model_dir / "vocab_auto.json",
            model_dir / "vocab_char.json",
        ]

        for candidate in candidates:
            if candidate and Path(candidate).exists():
                return str(candidate)

        return None

    @property
    def detector(self):
        """Lazy-load detector"""
        if self._detector is None:
            from .detector import TextDetector

            det_path = self.det_model_path
            # If no detector path specified, and we have a repo_id, and using DB/CRAFT
            if det_path is None and self.repo_id and self.det_method in ["db", "craft"]:
                det_path = self.repo_id

            self._detector = TextDetector(
                method=self.det_method,
                model_path=det_path,
                conf_threshold=self.det_conf_threshold,
            )
        return self._detector

    def _preprocess_region(self, img, box, extra_padding=5):
        """Preprocess a cropped region for recognition"""
        img_h, img_w = img.shape[:2]
        x, y, w, h = box

        # Add padding
        x1 = max(0, x - extra_padding)
        y1 = max(0, y - extra_padding)
        x2 = min(img_w, x + w + extra_padding)
        y2 = min(img_h, y + h + extra_padding)

        roi = img[y1:y2, x1:x2]

        if roi.size == 0:
            return None

        # Ensure grayscale
        if len(roi.shape) == 3:
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Invert if dark background
        if np.mean(roi) < 127:
            roi = 255 - roi

        roi_pil = Image.fromarray(roi)
        return preprocess_pil(self.cfg, roi_pil)

    def recognize_single_line_image(self, image_path):
        """Recognize text from a single-line image (no detection)"""
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")

        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if np.mean(img) < 127:
            img = 255 - img

        img_pil = Image.fromarray(img)
        img_tensor = preprocess_pil(self.cfg, img_pil)

        return self.recognize_region(img_tensor)

    def recognize_region(self, image_tensor):
        """
        Recognize text in a preprocessed region.

        Returns:
            (text, confidence) tuple
        """
        image_tensor = image_tensor.to(self.device)

        if self.cfg.USE_FP16 and self.device == "cuda":
            image_tensor = image_tensor.half()

        # Encode
        mem = self.model.encode(image_tensor)
        mem_proj = self.model.mem_proj(mem)

        # Get CTC logits for fusion
        ctc_logits = None
        if self.cfg.USE_CTC and hasattr(self.model, "ctc_head"):
            ctc_logits = self.model.ctc_head(mem)

        if self.use_beam_search:
            # Beam search (higher quality, slower)
            text, confidence = beam_decode_one_batched(
                self.model, mem_proj, self.tokenizer, self.cfg, ctc_logits_1=ctc_logits
            )
        else:
            # Greedy CTC (faster, simpler)
            text, confidence = greedy_ctc_decode(
                self.model, image_tensor, self.tokenizer, self.cfg
            )

        return text, confidence

    def process_document(self, image_path, mode="lines", verbose=False):
        """
        Process entire document with detection + recognition.

        Returns:
            List of dicts with 'box', 'text', 'confidence', etc.
        """
        if verbose:
            print(f"\n📄 Processing: {image_path}")
            print(f"🔲 Box padding: {self.padding}px")

        # Detect regions
        if mode == "lines":
            if hasattr(self.detector, "detect_lines_objects"):
                text_boxes = self.detector.detect_lines_objects(image_path)
                boxes = [b.bbox for b in text_boxes]
                det_confs = [b.confidence for b in text_boxes]
            else:
                boxes = self.detector.detect_lines(image_path)
                det_confs = [1.0] * len(boxes)
        else:
            boxes = self.detector.detect_words(image_path)
            det_confs = [1.0] * len(boxes)

        if verbose:
            print(f"🔍 Detected {len(boxes)} regions")

        # Load image
        img = cv2.imread(str(image_path))
        if len(img.shape) == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img

        # Recognize each region
        results = []
        for i, (box, det_conf) in enumerate(zip(boxes, det_confs), 1):
            try:
                region_tensor = self._preprocess_region(img_gray, box, extra_padding=5)
                if region_tensor is None:
                    continue

                text, confidence = self.recognize_region(region_tensor)

                results.append(
                    {
                        "box": [int(v) for v in box],
                        "text": text,
                        "confidence": float(confidence),
                        "det_confidence": float(det_conf),
                        "line_number": i,
                    }
                )

                if verbose:
                    print(f"  {i:2d}. {text[:50]:50s} ({confidence*100:.1f}%)")

            except Exception as e:
                if verbose:
                    print(f"  {i:2d}. [Error: {e}]")

        return results

    def extract_text(self, image_path, mode="lines", verbose=False):
        """Extract all text from document as string"""
        results = self.process_document(image_path, mode, verbose=verbose)

        if not results:
            return "", results

        # Sort by Y then X for reading order
        results.sort(key=lambda r: (r["box"][1], r["box"][0]))

        lines = []
        current_line = []
        prev_y = None
        prev_h = None

        for res in results:
            y, h = res["box"][1], res["box"][3]
            center_y = y + h / 2

            if prev_y is not None:
                prev_center = prev_y + prev_h / 2
                # Same line if centers are close
                if abs(center_y - prev_center) < max(h, prev_h) / 2:
                    current_line.append(res["text"])
                else:
                    lines.append(" ".join(current_line))
                    current_line = [res["text"]]
            else:
                current_line = [res["text"]]

            prev_y, prev_h = y, h

        if current_line:
            lines.append(" ".join(current_line))

        full_text = "\n".join(lines)
        return full_text, results
