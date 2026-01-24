# kiri_ocr/core.py
import torch
import cv2
import numpy as np
import sys
from PIL import Image
from pathlib import Path
import json

from .model import LightweightOCR, CharacterSet
from .model_transformer import (
    KiriOCR,
    CFG,
    CharTokenizer,
    beam_decode_one_batched,
    greedy_ctc_decode,
    preprocess_pil,
)


class OCR:
    """Complete Document OCR System with Padding"""

    _model_cache = {}

    def __init__(
        self,
        model_path="mrrtmob/kiri-ocr",
        det_model_path=None,
        det_method="db",
        det_conf_threshold=0.5,
        charset_path="models/charset_lite.txt",
        language="mixed",
        padding=10,
        device="cpu",
        verbose=False,
        use_beam_search=True,
        use_fp16=None,
    ):  # NEW: option to disable beam search
        """
        Args:
            model_path: Path to trained model (.kiri or .pth) or HuggingFace repo
            det_model_path: Path to YOLO detector model (optional)
            det_conf_threshold: Confidence threshold for YOLO detector
            charset_path: Path to character set (for legacy models)
            language: 'english', 'khmer', or 'mixed'
            padding: Pixels to pad around detected boxes
            device: 'cpu' or 'cuda'
            verbose: Print loading info
            use_beam_search: Use beam search (True) or greedy CTC (False)
            use_fp16: Force FP16 usage (True/False/None)
        """
        self.device = device
        self.verbose = verbose
        self.language = language
        self.padding = padding
        self.det_model_path = det_model_path
        self.det_method = det_method
        self.det_conf_threshold = det_conf_threshold
        self.use_beam_search = use_beam_search
        self.use_fp16 = use_fp16

        # Transformer-specific
        self.is_transformer = False
        self.transformer_cfg = None
        self.transformer_tok = None

        # Legacy model
        self.charset = None

        # Store repo_id for detector lazy loading
        self.repo_id = None
        if "/" in model_path and not model_path.startswith((".", "/")):
            self.repo_id = model_path

        # Resolve model path (HuggingFace, local, etc.)
        model_path = self._resolve_model_path(model_path)

        # Load model
        self._load_model(model_path, charset_path)

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

                # Try downloading vocab.json (user requested) or vocab_auto.json
                try:
                    hf_hub_download(repo_id=model_path, filename="vocab.json")
                except:
                    try:
                        hf_hub_download(repo_id=model_path, filename="vocab_auto.json")
                    except:
                        pass

                # Try downloading model.pt (user requested) or fallback to model.kiri
                try:
                    return hf_hub_download(repo_id=model_path, filename="model.pt")
                except:
                    return hf_hub_download(repo_id=model_path, filename="model.kiri")

            except Exception as e:
                if self.verbose:
                    print(f"⚠️ HuggingFace download failed: {e}")

        return model_path  # Return as-is, let load fail with clear error

    def _load_model(self, model_path, charset_path):
        """Load model from checkpoint"""
        cache_key = (str(model_path), self.device)

        if cache_key in OCR._model_cache:
            if self.verbose:
                print(f"⚡ Loading from memory cache")
            cached = OCR._model_cache[cache_key]
            self.model = cached["model"]
            self.is_transformer = cached["is_transformer"]
            if self.is_transformer:
                self.transformer_cfg = cached["cfg"]
                self.transformer_tok = cached["tok"]
            else:
                self.charset = cached["charset"]
            return

        if self.verbose:
            print(f"📦 Loading OCR model from {model_path}...")

        try:
            checkpoint = torch.load(
                model_path, map_location=self.device, weights_only=False
            )

            # Detect model type
            if self._is_transformer_checkpoint(checkpoint):
                self._load_transformer_model(checkpoint, model_path)
            else:
                self._load_legacy_model(checkpoint, charset_path)

            # Cache
            OCR._model_cache[cache_key] = {
                "model": self.model,
                "is_transformer": self.is_transformer,
                "cfg": self.transformer_cfg,
                "tok": self.transformer_tok,
                "charset": self.charset,
            }

        except RuntimeError as e:
            if "size mismatch" in str(e):
                print(f"\n❌ Model/charset size mismatch: {e}")
                sys.exit(1)
            raise

    def _is_transformer_checkpoint(self, checkpoint):
        """Check if checkpoint is transformer model"""
        if isinstance(checkpoint, dict):
            # Explicit transformer checkpoint
            if "config" in checkpoint and "vocab_path" in checkpoint:
                return True
            # Check for transformer layer names
            keys = (
                checkpoint.get("model", checkpoint).keys()
                if isinstance(checkpoint.get("model", checkpoint), dict)
                else checkpoint.keys()
            )
            return any(k.startswith(("stem.", "enc.", "pos2d.")) for k in keys)
        return False

    def _load_transformer_model(self, checkpoint, model_path):
        """Load transformer model"""
        if self.verbose:
            print(f"  ✓ Detected Transformer Model")

        self.is_transformer = True

        # Extract config and state dict
        if "config" in checkpoint:
            self.transformer_cfg = checkpoint["config"]
            state_dict = checkpoint["model"]
            vocab_path = checkpoint.get("vocab_path", "")
        else:
            self.transformer_cfg = CFG()
            state_dict = checkpoint
            vocab_path = ""

        # Override FP16 setting if requested
        if self.use_fp16 is not None:
            self.transformer_cfg.USE_FP16 = self.use_fp16

        # Find vocab file
        vocab_path = self._find_vocab_file(vocab_path, model_path)

        if not vocab_path or not Path(vocab_path).exists():
            raise FileNotFoundError(
                f"Could not find vocabulary file for transformer model. "
                f"Expected near: {model_path}"
            )

        self.transformer_tok = CharTokenizer(vocab_path, self.transformer_cfg)
        self.model = KiriOCR(self.transformer_cfg, self.transformer_tok).to(self.device)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()

        if self.transformer_cfg.USE_FP16 and self.device == "cuda":
            self.model.half()

        if self.verbose:
            print(f"  ✓ Loaded (Vocab: {self.transformer_tok.vocab_size} chars)")

    def _find_vocab_file(self, vocab_path, model_path):
        """Find vocabulary file for transformer model"""
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

    def _load_legacy_model(self, checkpoint, charset_path):
        """Load legacy LightweightOCR model"""
        self.is_transformer = False

        # Load charset
        if "charset" in checkpoint:
            if self.verbose:
                print(
                    f"  ✓ Found embedded charset ({len(checkpoint['charset'])} chars)"
                )
            self.charset = CharacterSet.from_checkpoint(checkpoint)
        else:
            charset_file = Path(charset_path)
            if not charset_file.exists():
                pkg_dir = Path(__file__).parent
                if (pkg_dir / charset_path).exists():
                    charset_path = str(pkg_dir / charset_path)

            if self.verbose:
                print(f"  ℹ️ Loading charset from: {charset_path}")
            self.charset = CharacterSet.load(charset_path)

        # Load model
        self.model = LightweightOCR(num_chars=len(self.charset)).eval()

        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)

        self.model = self.model.to(self.device)

        if self.verbose:
            print(f"  ✓ Legacy model loaded ({len(self.charset)} chars)")

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

        if self.is_transformer:
            return preprocess_pil(self.transformer_cfg, roi_pil)
        else:
            # Legacy preprocessing
            w, h = roi_pil.size
            new_h = 32
            new_w = max(32, int((w / h) * new_h))

            roi_pil = roi_pil.resize((new_w, new_h), Image.LANCZOS)
            roi_array = np.array(roi_pil) / 255.0
            return (
                torch.tensor(roi_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            )

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

        if self.is_transformer:
            img_tensor = preprocess_pil(self.transformer_cfg, img_pil)
        else:
            w, h = img_pil.size
            new_h = 32
            new_w = max(32, int((w / h) * new_h))
            img_pil = img_pil.resize((new_w, new_h), Image.LANCZOS)
            img_array = np.array(img_pil) / 255.0
            img_tensor = (
                torch.tensor(img_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            )

        return self.recognize_region(img_tensor)

    def recognize_region(self, image_tensor):
        """
        Recognize text in a preprocessed region.

        Returns:
            (text, confidence) tuple
        """
        image_tensor = image_tensor.to(self.device)

        if self.is_transformer:
            return self._recognize_transformer(image_tensor)
        else:
            return self._recognize_legacy(image_tensor)

    def _recognize_transformer(self, image_tensor):
        """Transformer model recognition"""
        cfg = self.transformer_cfg
        tok = self.transformer_tok

        if cfg.USE_FP16 and self.device == "cuda":
            image_tensor = image_tensor.half()

        # Encode
        mem = self.model.encode(image_tensor)
        mem_proj = self.model.mem_proj(mem)

        # Get CTC logits for fusion
        ctc_logits = None
        if cfg.USE_CTC and hasattr(self.model, "ctc_head"):
            ctc_logits = self.model.ctc_head(mem)

        if self.use_beam_search:
            # Beam search (higher quality, slower)
            text, confidence = beam_decode_one_batched(
                self.model, mem_proj, tok, cfg, ctc_logits_1=ctc_logits
            )
        else:
            # Greedy CTC (faster, simpler)
            text, confidence = greedy_ctc_decode(self.model, image_tensor, tok, cfg)

        return text, confidence

    def _recognize_legacy(self, image_tensor):
        """Legacy LightweightOCR recognition"""
        with torch.no_grad():
            logits = self.model(image_tensor)

            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1).squeeze().tolist()

            if not isinstance(preds, list):
                preds = [preds]

            # CTC decode with confidence
            char_confidences = []
            decoded_indices = []
            prev_idx = -1

            for i, idx in enumerate(preds):
                if idx != prev_idx:
                    if idx > 2:  # Skip BLANK, PAD, SOS
                        decoded_indices.append(idx)
                        # Get confidence for this timestep
                        conf = (
                            probs[i, 0, idx].item()
                            if probs.dim() == 3
                            else probs[i, idx].item()
                        )
                        char_confidences.append(conf)
                prev_idx = idx

            confidence = np.mean(char_confidences) if char_confidences else 0.0
            text = self.charset.decode(decoded_indices)

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
