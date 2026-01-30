"""
Core OCR class for Kiri OCR using Transformer architecture.

This module provides the main OCR class that combines text detection
and recognition into a unified document processing pipeline.
"""
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, Generator, List, Literal, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image

# Decoding method options
DecodeMethod = Literal["fast", "accurate", "beam", "ctc", "decoder"]

try:
    from safetensors.torch import load_file
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False

from .model import (
    CFG,
    CharTokenizer,
    KiriOCR,
    beam_decode_one_batched,
    greedy_ctc_decode,
    greedy_ctc_decode_streaming,
    greedy_decode_streaming,
    beam_decode_streaming,
    preprocess_pil,
)


class OCR:
    """
    Complete Document OCR System with Transformer Model.
    
    Combines text detection and recognition for full document OCR.
    Supports multiple detection backends (DB, CRAFT, legacy CV).
    
    Example:
        >>> ocr = OCR(model_path='mrrtmob/kiri-ocr')
        >>> text, results = ocr.extract_text('document.png')
        >>> print(text)
    """

    # Class-level model cache for memory efficiency
    _model_cache: Dict[Tuple[str, str], Dict] = {}

    def __init__(
        self,
        model_path: str = "mrrtmob/kiri-ocr",
        det_model_path: Optional[str] = None,
        det_method: str = "db",
        det_conf_threshold: float = 0.5,
        padding: int = 10,
        device: str = "cpu",
        verbose: bool = False,
        decode_method: DecodeMethod = "accurate",
        use_beam_search: Optional[bool] = None,  # Deprecated
        use_fp16: Optional[bool] = None,
    ):
        """
        Initialize the OCR system.

        Args:
            model_path: Path to recognition model (.safetensors/.pt) or HuggingFace repo ID
            det_model_path: Path to detection model (ONNX for DB, PTH for CRAFT)
            det_method: Detection method - 'db', 'craft', or 'legacy'
            det_conf_threshold: Confidence threshold for text detection
            padding: Pixels to pad around detected text regions
            device: Compute device - 'cpu' or 'cuda'
            verbose: Enable verbose output during processing
            decode_method: Text recognition decoding method:
                - 'fast' or 'ctc': Fast CTC decoding (no decoder, lower quality)
                - 'accurate' or 'decoder': Autoregressive decoder (better quality)
                - 'beam': Beam search decoder (best quality, slowest)
            use_beam_search: DEPRECATED - Use decode_method instead.
                            True maps to 'beam', False maps to 'fast'
            use_fp16: Force FP16 inference (None=auto, True/False=forced)
        """
        # Handle deprecated use_beam_search parameter
        if use_beam_search is not None:
            warnings.warn(
                "use_beam_search is deprecated. Use decode_method instead:\n"
                "  - decode_method='fast' (replaces use_beam_search=False)\n"
                "  - decode_method='accurate' (default, balanced)\n"
                "  - decode_method='beam' (replaces use_beam_search=True)",
                DeprecationWarning,
                stacklevel=2,
            )
            decode_method = "beam" if use_beam_search else "fast"
        
        # Normalize decode_method aliases
        decode_method = self._normalize_decode_method(decode_method)
        
        # Store configuration
        self.device = device
        self.verbose = verbose
        self.padding = padding
        self.det_model_path = det_model_path
        self.det_method = det_method
        self.det_conf_threshold = det_conf_threshold
        self.decode_method = decode_method
        self.use_fp16 = use_fp16
        
        # Keep use_beam_search for backward compatibility (derived from decode_method)
        self.use_beam_search = decode_method == "beam"

        # Model components (initialized in _load_model)
        self.cfg: Optional[CFG] = None
        self.tokenizer: Optional[CharTokenizer] = None
        self.model: Optional[KiriOCR] = None

        # Store repo_id for detector lazy loading (only for HuggingFace repos)
        self.repo_id: Optional[str] = None
        # A HuggingFace repo ID: contains "/" but is NOT a file path
        # File paths: start with ".", "/" or contain file extensions like ".safetensors", ".pt"
        is_likely_hf_repo = (
            "/" in model_path
            and not model_path.startswith((".", "/"))
            and not model_path.endswith((".safetensors", ".pt", ".onnx", ".pth"))
        )
        if is_likely_hf_repo:
            self.repo_id = model_path

        # Resolve and load model
        resolved_path = self._resolve_model_path(model_path)
        self._load_model(resolved_path)

        # Lazy-loaded detector
        self._detector = None

    @staticmethod
    def _normalize_decode_method(method: str) -> str:
        """Normalize decode method aliases to canonical names."""
        method = method.lower().strip()
        aliases = {
            "fast": "ctc",
            "ctc": "ctc",
            "accurate": "decoder",
            "decoder": "decoder",
            "beam": "beam",
        }
        if method not in aliases:
            raise ValueError(
                f"Invalid decode_method '{method}'. "
                f"Choose from: 'fast', 'accurate', 'beam' (or aliases: 'ctc', 'decoder')"
            )
        return aliases[method]

    # ==================== Model Loading ====================

    def _resolve_model_path(self, model_path: str) -> str:
        """
        Resolve model path from various sources.
        
        Checks in order:
        1. Direct path (if exists)
        2. Package directory
        3. HuggingFace Hub
        """
        model_file = Path(model_path)

        # Direct path
        if model_file.exists():
            return str(model_file)

        # Package directory fallback
        pkg_dir = Path(__file__).parent
        candidates = [
            pkg_dir / model_path,
            pkg_dir.parent / "models" / model_file.name,
        ]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)

        # HuggingFace Hub
        if "/" in model_path and not model_path.startswith((".", "/")):
            return self._download_from_huggingface(model_path)

        return model_path  # Return as-is, let load fail with clear error

    def _download_from_huggingface(self, repo_id: str) -> str:
        """Download model from HuggingFace Hub."""
        try:
            from huggingface_hub import hf_hub_download

            if self.verbose:
                print(f"⬇️ Downloading from HuggingFace: {repo_id}")

            # Download auxiliary files (optional)
            for filename in ["config.json", "vocab.json", "vocab_auto.json"]:
                try:
                    hf_hub_download(repo_id=repo_id, filename=filename)
                except Exception:
                    pass

            # Download model (prefer safetensors)
            for model_name in ["model.safetensors", "model.pt"]:
                try:
                    return hf_hub_download(repo_id=repo_id, filename=model_name)
                except Exception:
                    pass

        except Exception as e:
            if self.verbose:
                print(f"⚠️ HuggingFace download failed: {e}")

        return repo_id

    def _load_model(self, model_path: str) -> None:
        """Load Transformer model from checkpoint."""
        cache_key = (str(model_path), self.device)

        # Check cache
        if cache_key in OCR._model_cache:
            if self.verbose:
                print("⚡ Loading from memory cache")
            cached = OCR._model_cache[cache_key]
            self.model = cached["model"]
            self.cfg = cached["cfg"]
            self.tokenizer = cached["tokenizer"]
            return

        if self.verbose:
            print(f"📦 Loading OCR model from {model_path}...")

        try:
            # Load checkpoint based on format
            if model_path.endswith('.safetensors') and HAS_SAFETENSORS:
                state_dict, vocab_path = self._load_safetensors(model_path)
            else:
                state_dict, vocab_path = self._load_torch_checkpoint(model_path)

            # Override FP16 if requested
            if self.use_fp16 is not None:
                self.cfg.USE_FP16 = self.use_fp16

            # Find and load vocabulary
            vocab_path = self._find_vocab_file(vocab_path, model_path)
            if not vocab_path or not Path(vocab_path).exists():
                raise FileNotFoundError(
                    f"Could not find vocabulary file. Expected near: {model_path}"
                )

            # Check if model has decoder positional encoding (new architecture)
            has_dec_pos_enc = any("dec_pos_enc" in k for k in state_dict.keys())
            
            # Initialize model (disable pos enc for old models to avoid random weights)
            self.tokenizer = CharTokenizer(vocab_path, self.cfg)
            self.model = KiriOCR(
                self.cfg,
                self.tokenizer,
                use_dec_pos_enc=has_dec_pos_enc
            ).to(self.device)
            
            # Load state dict with strict=False for backward compatibility
            missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
            
            # Warn if decoder positional encoding is missing (old model)
            if not has_dec_pos_enc:
                if self.verbose:
                    print("  ⚠️ Old model without decoder positional encoding")
                    print("    Decoder quality may be limited. Consider retraining with new architecture.")
            
            self.model.eval()

            # Enable FP16 on CUDA
            if self.cfg.USE_FP16 and self.device == "cuda":
                self.model.half()

            if self.verbose:
                print(f"  ✓ Loaded (Vocab: {self.tokenizer.vocab_size} chars)")

            # Cache for future use
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

    def _load_safetensors(self, model_path: str) -> Tuple[Dict, str]:
        """Load model from safetensors format."""
        state_dict = load_file(model_path, device=self.device)
        
        # Initialize config
        self.cfg = CFG()
        vocab_path = ""
        
        # Load metadata if available
        metadata_path = model_path.replace('.safetensors', '_meta.json')
        if Path(metadata_path).exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            vocab_path = metadata.get("vocab_path", "")
            self._apply_config(metadata.get("config", {}))
        else:
            # Fallback: infer architecture from state_dict
            if self.verbose:
                print("  ⚠️ No metadata file found, inferring architecture from weights...")
            self._infer_config_from_state_dict(state_dict)
        
        return state_dict, vocab_path
    
    def _infer_config_from_state_dict(self, state_dict: Dict) -> None:
        """
        Infer model architecture from state_dict keys and shapes.
        
        This is a fallback for legacy models without metadata.
        """
        # Infer ENC_DIM from stem output channel (stem.net.9.weight is [enc_dim, 160, 3, 3])
        if "stem.net.9.weight" in state_dict:
            self.cfg.ENC_DIM = state_dict["stem.net.9.weight"].shape[0]
            if self.verbose:
                print(f"    Inferred ENC_DIM={self.cfg.ENC_DIM}")
        
        # Count encoder layers (enc.layers.0, enc.layers.1, ...)
        enc_layers = set()
        for key in state_dict.keys():
            if key.startswith("enc.layers."):
                layer_num = int(key.split(".")[2])
                enc_layers.add(layer_num)
        if enc_layers:
            self.cfg.ENC_LAYERS = max(enc_layers) + 1
            if self.verbose:
                print(f"    Inferred ENC_LAYERS={self.cfg.ENC_LAYERS}")
        
        # Count decoder layers (dec.layers.0, dec.layers.1, ...)
        dec_layers = set()
        for key in state_dict.keys():
            if key.startswith("dec.layers."):
                layer_num = int(key.split(".")[2])
                dec_layers.add(layer_num)
        if dec_layers:
            self.cfg.DEC_LAYERS = max(dec_layers) + 1
            if self.verbose:
                print(f"    Inferred DEC_LAYERS={self.cfg.DEC_LAYERS}")
        
        # Infer ENC_FF from encoder FFN weight shape
        for key in state_dict.keys():
            if "enc.layers.0.linear1.weight" in key:
                self.cfg.ENC_FF = state_dict[key].shape[0]
                if self.verbose:
                    print(f"    Inferred ENC_FF={self.cfg.ENC_FF}")
                break
        
        # Infer DEC_DIM from decoder embedding
        if "dec_emb.weight" in state_dict:
            self.cfg.DEC_DIM = state_dict["dec_emb.weight"].shape[1]
            if self.verbose:
                print(f"    Inferred DEC_DIM={self.cfg.DEC_DIM}")
        
        # Infer DEC_FF from decoder FFN weight shape
        for key in state_dict.keys():
            if "dec.layers.0.linear1.weight" in key:
                self.cfg.DEC_FF = state_dict[key].shape[0]
                if self.verbose:
                    print(f"    Inferred DEC_FF={self.cfg.DEC_FF}")
                break
        
        # Infer ENC_HEADS from attention in_proj_weight (shape is [3*dim, dim])
        # num_heads = dim / head_dim, typically head_dim=64
        for key in state_dict.keys():
            if "enc.layers.0.self_attn.in_proj_weight" in key:
                total_dim = state_dict[key].shape[0] // 3
                # Assume head_dim is 64 (common default)
                if total_dim % 64 == 0:
                    self.cfg.ENC_HEADS = total_dim // 64
                elif total_dim % 32 == 0:
                    self.cfg.ENC_HEADS = total_dim // 32
                else:
                    self.cfg.ENC_HEADS = 8  # fallback default
                if self.verbose:
                    print(f"    Inferred ENC_HEADS={self.cfg.ENC_HEADS}")
                break
        
        # Infer DEC_HEADS similarly
        for key in state_dict.keys():
            if "dec.layers.0.self_attn.in_proj_weight" in key:
                total_dim = state_dict[key].shape[0] // 3
                if total_dim % 64 == 0:
                    self.cfg.DEC_HEADS = total_dim // 64
                elif total_dim % 32 == 0:
                    self.cfg.DEC_HEADS = total_dim // 32
                else:
                    self.cfg.DEC_HEADS = 8  # fallback default
                if self.verbose:
                    print(f"    Inferred DEC_HEADS={self.cfg.DEC_HEADS}")
                break

    def _load_torch_checkpoint(self, model_path: str) -> Tuple[Dict, str]:
        """Load model from PyTorch checkpoint."""
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        if "config" in checkpoint:
            config_data = checkpoint["config"]
            if isinstance(config_data, dict):
                self.cfg = CFG()
                self._apply_config(config_data)
            else:
                self.cfg = config_data
            state_dict = checkpoint["model"]
            vocab_path = checkpoint.get("vocab_path", "")
        else:
            self.cfg = CFG()
            state_dict = checkpoint
            vocab_path = ""
        
        return state_dict, vocab_path

    def _apply_config(self, config_data: Dict) -> None:
        """Apply configuration dict to CFG object."""
        if not config_data:
            return
        # Image dimensions
        self.cfg.IMG_H = config_data.get("IMG_H", self.cfg.IMG_H)
        self.cfg.IMG_W = config_data.get("IMG_W", self.cfg.IMG_W)
        
        # Encoder architecture
        self.cfg.ENC_DIM = config_data.get("ENC_DIM", self.cfg.ENC_DIM)
        self.cfg.ENC_LAYERS = config_data.get("ENC_LAYERS", self.cfg.ENC_LAYERS)
        self.cfg.ENC_HEADS = config_data.get("ENC_HEADS", self.cfg.ENC_HEADS)
        self.cfg.ENC_FF = config_data.get("ENC_FF", self.cfg.ENC_FF)
        
        # Decoder architecture
        self.cfg.DEC_DIM = config_data.get("DEC_DIM", self.cfg.DEC_DIM)
        self.cfg.DEC_LAYERS = config_data.get("DEC_LAYERS", self.cfg.DEC_LAYERS)
        self.cfg.DEC_HEADS = config_data.get("DEC_HEADS", self.cfg.DEC_HEADS)
        self.cfg.DEC_FF = config_data.get("DEC_FF", self.cfg.DEC_FF)
        
        # Regularization
        self.cfg.DROPOUT = config_data.get("DROPOUT", self.cfg.DROPOUT)
        
        # Training flags
        self.cfg.USE_CTC = config_data.get("USE_CTC", self.cfg.USE_CTC)
        self.cfg.USE_FP16 = config_data.get("USE_FP16", self.cfg.USE_FP16)

    def _find_vocab_file(self, vocab_path: str, model_path: str) -> Optional[str]:
        """Find vocabulary file for the model."""
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

    # ==================== Detection ====================

    @property
    def detector(self):
        """Lazy-load text detector on first access."""
        if self._detector is None:
            from .detector import TextDetector

            det_path = self.det_model_path
            # Use repo_id for detector if not specified
            if det_path is None and self.repo_id and self.det_method in ["db", "craft"]:
                det_path = self.repo_id

            self._detector = TextDetector(
                method=self.det_method,
                model_path=det_path,
                conf_threshold=self.det_conf_threshold,
            )
        return self._detector

    # ==================== Recognition ====================

    def _preprocess_region(
        self, 
        img: np.ndarray, 
        box: Tuple[int, int, int, int], 
        extra_padding: int = 5
    ) -> Optional[torch.Tensor]:
        """
        Preprocess a cropped region for recognition.
        
        Args:
            img: Grayscale image array
            box: Bounding box (x, y, w, h)
            extra_padding: Additional padding around the box
            
        Returns:
            Preprocessed tensor ready for model input, or None if invalid
        """
        img_h, img_w = img.shape[:2]
        x, y, w, h = box

        # Apply padding with bounds checking
        x1 = max(0, x - extra_padding)
        y1 = max(0, y - extra_padding)
        x2 = min(img_w, x + w + extra_padding)
        y2 = min(img_h, y + h + extra_padding)

        roi = img[y1:y2, x1:x2]
        if roi.size == 0:
            return None

        # Convert to grayscale if needed
        if len(roi.shape) == 3:
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Invert if dark background (assume light text on dark)
        if np.mean(roi) < 127:
            roi = 255 - roi

        roi_pil = Image.fromarray(roi)
        return preprocess_pil(self.cfg, roi_pil)

    def recognize_region(self, image_tensor: torch.Tensor) -> Tuple[str, float]:
        """
        Recognize text in a preprocessed image tensor.

        Args:
            image_tensor: Preprocessed image tensor from preprocess_pil()

        Returns:
            Tuple of (recognized_text, confidence_score)
        """
        image_tensor = image_tensor.to(self.device)

        if self.cfg.USE_FP16 and self.device == "cuda":
            image_tensor = image_tensor.half()

        # Encode image
        mem = self.model.encode(image_tensor)
        mem_proj = self.model.mem_proj(mem)

        # Get CTC logits for hybrid decoding
        ctc_logits = None
        if self.cfg.USE_CTC and hasattr(self.model, "ctc_head"):
            ctc_logits = self.model.ctc_head(mem)

        # Decode based on method
        if self.decode_method == "ctc":
            # Fast CTC decoding (no decoder)
            text, confidence = greedy_ctc_decode(
                self.model, image_tensor, self.tokenizer, self.cfg
            )
        elif self.decode_method == "decoder":
            # Greedy decoder (balanced speed/quality)
            # Use beam search with beam=1 for greedy decoding
            from .model import beam_decode_one_batched
            old_beam = self.cfg.BEAM
            self.cfg.BEAM = 1
            text, confidence = beam_decode_one_batched(
                self.model, mem_proj, self.tokenizer, self.cfg, ctc_logits_1=ctc_logits
            )
            self.cfg.BEAM = old_beam
        else:  # beam
            # Full beam search (best quality)
            text, confidence = beam_decode_one_batched(
                self.model, mem_proj, self.tokenizer, self.cfg, ctc_logits_1=ctc_logits
            )

        return text, confidence

    def recognize_region_streaming(
        self,
        image_tensor: torch.Tensor,
        decode_method: Optional[str] = None,
    ) -> Generator[Dict, None, None]:
        """
        Recognize text with character-by-character streaming, like LLM generation.
        
        Yields each character/token as it's decoded by the model.
        
        Args:
            image_tensor: Preprocessed image tensor from preprocess_pil()
            decode_method: Decoding method override. Options:
                - 'fast' or 'ctc': Fast CTC decoding
                - 'accurate' or 'decoder': Greedy decoder
                - 'beam': Beam search decoder
                - None: Use instance setting (self.decode_method)
            
        Yields:
            Dict with:
            - 'token': The new character/token string
            - 'text': Full decoded text so far
            - 'confidence': Token/sequence confidence
            - 'step': Current decoding step
            - 'finished': Whether decoding is complete
            
        Example:
            >>> tensor = ocr._preprocess_region(img, box)
            >>> for result in ocr.recognize_region_streaming(tensor):
            ...     print(result['token'], end='', flush=True)
            ...     if result['finished']:
            ...         print()
        """
        image_tensor = image_tensor.to(self.device)

        if self.cfg.USE_FP16 and self.device == "cuda":
            image_tensor = image_tensor.half()

        # Encode image
        mem = self.model.encode(image_tensor)
        mem_proj = self.model.mem_proj(mem)

        # Get CTC logits for length estimation
        ctc_logits = None
        if self.cfg.USE_CTC and hasattr(self.model, "ctc_head"):
            ctc_logits = self.model.ctc_head(mem)

        # Use instance setting if not explicitly specified
        method = decode_method
        if method is not None:
            method = self._normalize_decode_method(method)
        else:
            method = self.decode_method

        # Stream decode based on method
        if method == "ctc":
            # CTC streaming (fast, no decoder)
            yield from greedy_ctc_decode_streaming(
                self.model, mem, self.tokenizer, self.cfg
            )
        elif method == "decoder":
            # Greedy decoder streaming
            yield from greedy_decode_streaming(
                self.model, mem_proj, self.tokenizer, self.cfg, ctc_logits_1=ctc_logits
            )
        else:  # beam
            # Beam search streaming
            yield from beam_decode_streaming(
                self.model, mem_proj, self.tokenizer, self.cfg, ctc_logits_1=ctc_logits
            )

    def recognize_streaming(
        self,
        image_path: Union[str, Path],
        decode_method: Optional[str] = None,
    ) -> Generator[Dict, None, None]:
        """
        Recognize text from a single-line image with character streaming.
        
        Like LLM text generation - yields each character as decoded.
        
        Args:
            image_path: Path to the single-line text image
            decode_method: Decoding method override ('fast', 'accurate', 'beam').
                          None = use instance setting
            
        Yields:
            Dict with token, text, confidence, step, finished
            
        Example:
            >>> for chunk in ocr.recognize_streaming('line.png'):
            ...     print(chunk['token'], end='', flush=True)
        """
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")

        # Convert to grayscale
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Invert if dark background
        if np.mean(img) < 127:
            img = 255 - img

        img_pil = Image.fromarray(img)
        img_tensor = preprocess_pil(self.cfg, img_pil)

        yield from self.recognize_region_streaming(img_tensor, decode_method)

    def recognize_single_line_image(
        self, 
        image_path: Union[str, Path]
    ) -> Tuple[str, float]:
        """
        Recognize text from a single-line image without detection.

        Args:
            image_path: Path to the image file

        Returns:
            Tuple of (recognized_text, confidence_score)
        """
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")

        # Convert to grayscale
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Invert if dark background
        if np.mean(img) < 127:
            img = 255 - img

        img_pil = Image.fromarray(img)
        img_tensor = preprocess_pil(self.cfg, img_pil)

        return self.recognize_region(img_tensor)

    # ==================== Document Processing ====================

    def process_document(
        self,
        image_path: Union[str, Path],
        mode: str = "lines",
        verbose: bool = False,
    ) -> List[Dict]:
        """
        Process a document image with detection and recognition.

        Args:
            image_path: Path to the document image
            mode: Detection mode - 'lines' or 'words'
            verbose: Enable verbose output

        Returns:
            List of result dicts with keys:
            - 'box': [x, y, w, h] bounding box
            - 'text': Recognized text string
            - 'confidence': Recognition confidence (0-1)
            - 'det_confidence': Detection confidence (0-1)
            - 'line_number': Sequential line number
        """
        if verbose:
            print(f"\n📄 Processing: {image_path}")
            print(f"🔲 Box padding: {self.padding}px")

        # Detect text regions
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

        # Load and prepare image
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
            
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

        # Recognize each region
        results = []
        for i, (box, det_conf) in enumerate(zip(boxes, det_confs), 1):
            try:
                region_tensor = self._preprocess_region(img_gray, box, extra_padding=5)
                if region_tensor is None:
                    continue

                text, confidence = self.recognize_region(region_tensor)

                results.append({
                    "box": [int(v) for v in box],
                    "text": text,
                    "confidence": float(confidence),
                    "det_confidence": float(det_conf),
                    "line_number": i,
                })

                if verbose:
                    print(f"  {i:2d}. {text[:50]:50s} ({confidence*100:.1f}%)")

            except Exception as e:
                if verbose:
                    print(f"  {i:2d}. [Error: {e}]")

        return results

    def process_document_streaming(
        self,
        image_path: Union[str, Path],
        mode: str = "lines",
        verbose: bool = False,
    ) -> Generator[Dict, None, None]:
        """
        Process a document image with streaming/real-time inference.
        
        Yields each text region result as soon as it's recognized,
        without waiting for all regions to be processed.

        Args:
            image_path: Path to the document image
            mode: Detection mode - 'lines' or 'words'
            verbose: Enable verbose output

        Yields:
            Dict for each recognized region with keys:
            - 'box': [x, y, w, h] bounding box
            - 'text': Recognized text string
            - 'confidence': Recognition confidence (0-1)
            - 'det_confidence': Detection confidence (0-1)
            - 'line_number': Sequential line number
            - 'total_regions': Total number of detected regions
        """
        if verbose:
            print(f"\n📄 Processing (streaming): {image_path}")
            print(f"🔲 Box padding: {self.padding}px")

        # Detect text regions
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

        total_regions = len(boxes)
        
        if verbose:
            print(f"🔍 Detected {total_regions} regions")

        # Load and prepare image
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
            
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

        # Recognize and yield each region immediately
        for i, (box, det_conf) in enumerate(zip(boxes, det_confs), 1):
            try:
                region_tensor = self._preprocess_region(img_gray, box, extra_padding=5)
                if region_tensor is None:
                    continue

                text, confidence = self.recognize_region(region_tensor)

                result = {
                    "box": [int(v) for v in box],
                    "text": text,
                    "confidence": float(confidence),
                    "det_confidence": float(det_conf),
                    "line_number": i,
                    "total_regions": total_regions,
                }

                if verbose:
                    print(f"  {i:2d}. {text[:50]:50s} ({confidence*100:.1f}%)")

                yield result

            except Exception as e:
                if verbose:
                    print(f"  {i:2d}. [Error: {e}]")
                # Yield error result so caller knows about failures
                yield {
                    "box": [int(v) for v in box],
                    "text": "",
                    "confidence": 0.0,
                    "det_confidence": float(det_conf),
                    "line_number": i,
                    "total_regions": total_regions,
                    "error": str(e),
                }

    def extract_text_stream_chars(
        self,
        image_path: Union[str, Path],
        mode: str = "lines",
        decode_method: Optional[str] = None,
        verbose: bool = False,
    ) -> Generator[Dict, None, None]:
        """
        Extract text with LLM-style character-by-character streaming.
        
        For each detected text region, yields each character as it's decoded,
        just like LLM text generation. Perfect for real-time display.
        
        Args:
            image_path: Path to the document image
            mode: Detection mode - 'lines' or 'words'
            decode_method: Decoding method override ('fast', 'accurate', 'beam').
                          None = use instance setting (self.decode_method)
            verbose: Enable verbose output

        Yields:
            Dict with:
            - 'token': New character/token (empty string at region boundaries)
            - 'text': Full text of current region so far
            - 'cumulative_text': All text from all regions so far
            - 'region_number': Current region being processed
            - 'total_regions': Total detected regions
            - 'step': Decoding step within current region
            - 'region_finished': Whether current region is done
            - 'document_finished': Whether all regions are done
            - 'region_start': True if this is the start of a new region
            - 'box': Bounding box of current region
            
        Example:
            >>> for chunk in ocr.extract_text_stream_chars('document.png'):
            ...     if chunk['region_start']:
            ...         print(f"\\nRegion {chunk['region_number']}: ", end='')
            ...     print(chunk['token'], end='', flush=True)
        """
        if verbose:
            print(f"\n📄 Processing (char streaming): {image_path}")
        
        # Detect text regions
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
        
        total_regions = len(boxes)
        
        if verbose:
            print(f"🔍 Detected {total_regions} regions")
        
        # Load image
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        
        all_region_texts = []
        
        for region_num, (box, det_conf) in enumerate(zip(boxes, det_confs), 1):
            try:
                region_tensor = self._preprocess_region(img_gray, box, extra_padding=5)
                if region_tensor is None:
                    continue
                
                # Yield region start marker
                yield {
                    "token": "",
                    "text": "",
                    "cumulative_text": "\n".join(all_region_texts),
                    "region_number": region_num,
                    "total_regions": total_regions,
                    "step": 0,
                    "region_finished": False,
                    "document_finished": False,
                    "region_start": True,
                    "box": [int(v) for v in box],
                    "det_confidence": float(det_conf),
                }
                
                current_region_text = ""
                
                # Stream characters for this region
                for chunk in self.recognize_region_streaming(region_tensor, decode_method):
                    current_region_text = chunk["text"]
                    
                    # Build cumulative text
                    temp_texts = all_region_texts + ([current_region_text] if current_region_text else [])
                    cumulative = "\n".join(temp_texts)
                    
                    yield {
                        "token": chunk["token"],
                        "text": current_region_text,
                        "cumulative_text": cumulative,
                        "region_number": region_num,
                        "total_regions": total_regions,
                        "step": chunk["step"],
                        "confidence": chunk["confidence"],
                        "region_finished": chunk["finished"],
                        "document_finished": chunk["finished"] and region_num == total_regions,
                        "region_start": False,
                        "box": [int(v) for v in box],
                        "det_confidence": float(det_conf),
                    }
                    
                    if chunk["finished"]:
                        break
                
                if current_region_text:
                    all_region_texts.append(current_region_text)
                    
                if verbose:
                    print(f"  {region_num:2d}. {current_region_text[:50]}")
                    
            except Exception as e:
                if verbose:
                    print(f"  {region_num:2d}. [Error: {e}]")
                yield {
                    "token": "",
                    "text": "",
                    "cumulative_text": "\n".join(all_region_texts),
                    "region_number": region_num,
                    "total_regions": total_regions,
                    "step": 0,
                    "region_finished": True,
                    "document_finished": region_num == total_regions,
                    "region_start": True,
                    "box": [int(v) for v in box],
                    "error": str(e),
                }

    def extract_text_streaming(
        self,
        image_path: Union[str, Path],
        mode: str = "lines",
        verbose: bool = False,
    ) -> Generator[Dict, None, None]:
        """
        Extract text with real-time streaming - yields each region as it's recognized.
        
        This is useful for:
        - Real-time UI updates showing progress
        - Processing results before full document is complete
        - Displaying text character by character as recognized
        
        Args:
            image_path: Path to the document image
            mode: Detection mode - 'lines' or 'words'
            verbose: Enable verbose output

        Yields:
            Dict for each region containing:
            - 'box': [x, y, w, h] bounding box
            - 'text': Recognized text string
            - 'confidence': Recognition confidence (0-1)
            - 'det_confidence': Detection confidence (0-1)
            - 'line_number': Current region number
            - 'total_regions': Total number of detected regions
            - 'cumulative_text': All text recognized so far (lines joined)
            
        Example:
            >>> ocr = OCR()
            >>> for result in ocr.extract_text_streaming('document.png'):
            ...     print(f"Region {result['line_number']}/{result['total_regions']}")
            ...     print(f"Text: {result['text']}")
            ...     print(f"Progress: {result['cumulative_text']}")
        """
        all_results = []
        lines = []
        current_line = []
        prev_center_y = None
        prev_height = None
        
        for result in self.process_document_streaming(image_path, mode, verbose):
            all_results.append(result)
            
            if "error" not in result and result["text"]:
                # Group into lines based on vertical position
                y, h = result["box"][1], result["box"][3]
                center_y = y + h / 2

                if prev_center_y is not None:
                    tolerance = max(h, prev_height) * 0.8
                    
                    if abs(center_y - prev_center_y) < tolerance:
                        current_line.append(result["text"])
                    else:
                        if current_line:
                            lines.append(" ".join(current_line))
                        current_line = [result["text"]]
                else:
                    current_line = [result["text"]]

                prev_center_y = center_y
                prev_height = h
            
            # Build cumulative text including current line
            temp_lines = lines.copy()
            if current_line:
                temp_lines.append(" ".join(current_line))
            cumulative_text = "\n".join(temp_lines)
            
            # Add cumulative text to result
            result["cumulative_text"] = cumulative_text
            
            yield result

    def extract_text(
        self,
        image_path: Union[str, Path],
        mode: str = "lines",
        verbose: bool = False,
    ) -> Tuple[str, List[Dict]]:
        """
        Extract all text from a document image.

        Args:
            image_path: Path to the document image
            mode: Detection mode - 'lines' or 'words'
            verbose: Enable verbose output

        Returns:
            Tuple of:
            - Full extracted text as string (lines joined by newlines)
            - List of detailed result dicts from process_document()
        """
        results = self.process_document(image_path, mode, verbose=verbose)

        if not results:
            return "", results

        # Group results into text lines based on vertical position
        # Results are already sorted by detector in reading order
        lines = []
        current_line = []
        prev_center_y = None
        prev_height = None

        for res in results:
            y, h = res["box"][1], res["box"][3]
            center_y = y + h / 2

            if prev_center_y is not None:
                # Use 80% of max height as same-line tolerance
                tolerance = max(h, prev_height) * 0.8
                
                if abs(center_y - prev_center_y) < tolerance:
                    # Same line - append text
                    current_line.append(res["text"])
                else:
                    # New line - save current and start new
                    lines.append(" ".join(current_line))
                    current_line = [res["text"]]
            else:
                current_line = [res["text"]]

            prev_center_y = center_y
            prev_height = h

        # Don't forget the last line
        if current_line:
            lines.append(" ".join(current_line))

        full_text = "\n".join(lines)
        return full_text, results
