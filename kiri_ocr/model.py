"""
KiriOCR Transformer Model.

This is the main OCR model using a hybrid architecture with:
- CNN backbone for visual feature extraction
- Transformer encoder for contextual understanding  
- CTC head for fast alignment-free decoding
- Attention decoder for accurate sequence generation

For the legacy CRNN model, see kiri_ocr.legacy.model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Generator, List, Dict, Optional, Tuple, NamedTuple
import json
import math
import numpy as np
from PIL import Image


# ========== CONFIGURATION ==========
@dataclass
class CFG:
    # --- Model Architecture ---
    IMG_H: int = 48
    IMG_W: int = 640
    MAX_DEC_LEN: int = 512
    UNK_TOKEN: str = "<unk>"
    COLLAPSE_WHITESPACE: bool = True
    UNICODE_NFC: bool = True

    ENC_DIM: int = 256
    ENC_LAYERS: int = 4
    ENC_HEADS: int = 8
    ENC_FF: int = 1024
    DROPOUT: float = 0.15

    USE_DECODER: bool = True
    DEC_DIM: int = 256
    DEC_LAYERS: int = 3
    DEC_HEADS: int = 8
    DEC_FF: int = 1024

    USE_CTC: bool = True
    USE_LM: bool = True
    USE_LM_FUSION_EVAL: bool = True
    LM_FUSION_ALPHA: float = 0.35
    USE_FP16: bool = True
    USE_AUTOCAST: bool = True

    # --- Inference Params ---
    CTC_FUSION_ALPHA: float = 0.35
    BEAM: int = 4
    BEAM_LENP: float = 0.75

    EOS_LOGP_BIAS: float = 0.55
    EOS_LOGP_BOOST: float = 0.65
    EOS_BIAS_UNTIL_LEN: int = 28

    REPEAT_LAST_PENALTY: float = 2.0      # Penalty for exact token repeat
    REPEAT_BIGRAM_PENALTY: float = 1.5    # Penalty for bi-gram repeat (e.g. AB-AB)
    REPEAT_TRIGRAM_PENALTY: float = 1.2   # Penalty for tri-gram repeat
    UNK_LOGP_PENALTY: float = 1.0

    DEC_MAX_LEN_RATIO: float = 1.35
    DEC_MAX_LEN_PAD: int = 6
    MEM_MAX_LEN_RATIO: float = 0.75


# ========== RESULT TYPE ==========
class OCRResult(NamedTuple):
    """Structured result from OCR inference"""

    text: str
    confidence: float
    ctc_confidence: Optional[float] = None
    decoder_confidence: Optional[float] = None


# ========== TOKENIZER ==========
class CharTokenizer:
    def __init__(self, vocab_path: str, cfg: CFG):
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab_raw: Dict[str, int] = json.load(f)

        if cfg.UNK_TOKEN not in vocab_raw:
            vocab_raw[cfg.UNK_TOKEN] = max(vocab_raw.values(), default=-1) + 1

        items = sorted(vocab_raw.items(), key=lambda kv: kv[1])
        self.token_to_id = {tok: i for i, (tok, _) in enumerate(items)}
        self.id_to_token = {i: tok for i, (tok, _) in enumerate(items)}

        self.unk_token = cfg.UNK_TOKEN
        self.unk_id = self.token_to_id[cfg.UNK_TOKEN]
        self.blank_id = 0
        self.pad_id = 1
        self.ctc_offset = 2
        self.vocab_size = len(self.token_to_id)
        self.ctc_classes = self.vocab_size + self.ctc_offset

        self.dec_pad = 0
        self.dec_bos = 1
        self.dec_eos = 2
        self.dec_offset = 3
        self.dec_vocab = self.vocab_size + self.dec_offset

    def decode_ctc(self, ids: List[int]) -> str:
        """Decode CTC output with deduplication"""
        chars = []
        prev_id = None
        for idx in ids:
            if idx == prev_id:
                continue
            prev_id = idx
            if idx < self.ctc_offset:
                continue
            raw_id = idx - self.ctc_offset
            if 0 <= raw_id < self.vocab_size:
                char = self.id_to_token.get(raw_id, "")
                if char != self.unk_token:
                    chars.append(char)
        return "".join(chars)

    def decode_dec(self, ids: List[int]) -> str:
        out = []
        for x in ids:
            if x in (self.dec_pad, self.dec_bos, self.dec_eos):
                continue
            y = x - self.dec_offset
            if 0 <= y < self.vocab_size:
                t = self.id_to_token.get(y, self.unk_token)
                out.append("" if t == self.unk_token else t)
        return "".join(out)

    def dec_to_ctc_id(self, dec_id: int) -> int:
        """Convert decoder token ID to CTC token ID"""
        if dec_id in (self.dec_pad, self.dec_bos, self.dec_eos):
            return self.blank_id
        raw_id = dec_id - self.dec_offset
        if 0 <= raw_id < self.vocab_size:
            return raw_id + self.ctc_offset
        return self.unk_id + self.ctc_offset


# ========== MODEL COMPONENTS ==========
class SinusoidalPosEnc1D(nn.Module):
    """Sinusoidal positional encoding for 1D sequences (decoder)"""
    def __init__(self, dim: int, max_len: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding buffer
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, dim]
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, dim]
        Returns:
            x + positional encoding
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class PosEnc2D(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def _make_pe(
        self, length: int, dim: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        pos = torch.arange(length, dtype=dtype, device=device).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, dim, 2, dtype=dtype, device=device)
            * (-math.log(10000.0) / dim)
        )
        pe = torch.zeros((length, dim), dtype=dtype, device=device)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        num_feats = c // 2
        if num_feats == 0:
            return x
        pe_y = self._make_pe(h, num_feats, x.device, x.dtype)
        pe_x = self._make_pe(w, num_feats, x.device, x.dtype)
        pe_y = pe_y.unsqueeze(2).repeat(1, 1, w)
        pe_x = pe_x.transpose(0, 1).unsqueeze(0).repeat(h, 1, 1)
        pe = torch.cat([pe_y, pe_x], dim=1)
        pe = pe.permute(1, 0, 2)
        if pe.size(0) < c:
            pad = torch.zeros((c - pe.size(0), h, w), device=x.device, dtype=x.dtype)
            pe = torch.cat([pe, pad], dim=0)
        return x + pe.unsqueeze(0)


class ConvStem(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 48, 3, 1, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.SiLU(inplace=True),
            nn.Conv2d(48, 96, 3, (2, 2), 1, bias=False),
            nn.BatchNorm2d(96),
            nn.SiLU(inplace=True),
            nn.Conv2d(96, 160, 3, (2, 2), 1, bias=False),
            nn.BatchNorm2d(160),
            nn.SiLU(inplace=True),
            nn.Conv2d(160, dim, 3, (2, 1), 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU(inplace=True),
            nn.Dropout2d(dropout),
        )

    def forward(self, x):
        return self.net(x)


# ========== MAIN MODEL ==========
class KiriOCR(nn.Module):
    def __init__(self, cfg: CFG, tok: CharTokenizer, use_dec_pos_enc: bool = True):
        super().__init__()
        self.cfg = cfg
        self.tok = tok
        self.use_dec_pos_enc = use_dec_pos_enc  # Flag to control decoder positional encoding
        d = cfg.DROPOUT

        self.stem = ConvStem(cfg.ENC_DIM, d)
        self.pos2d = PosEnc2D(cfg.ENC_DIM)

        self.enc_ln_in = nn.LayerNorm(cfg.ENC_DIM)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.ENC_DIM,
            nhead=cfg.ENC_HEADS,
            dim_feedforward=cfg.ENC_FF,
            dropout=d,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.enc = nn.TransformerEncoder(
            enc_layer,
            num_layers=cfg.ENC_LAYERS,
            enable_nested_tensor=False,
        )
        self.enc_ln = nn.LayerNorm(cfg.ENC_DIM)

        if cfg.USE_CTC:
            self.ctc_head = nn.Sequential(
                nn.LayerNorm(cfg.ENC_DIM),
                nn.Dropout(d),
                nn.Linear(cfg.ENC_DIM, tok.ctc_classes),
            )

        self.mem_proj = nn.Linear(cfg.ENC_DIM, cfg.DEC_DIM, bias=False)
        self.dec_emb = nn.Embedding(tok.dec_vocab, cfg.DEC_DIM)
        
        # Positional encoding for decoder (can be disabled for old models)
        if use_dec_pos_enc:
            self.dec_pos_enc = SinusoidalPosEnc1D(
                dim=cfg.DEC_DIM,
                max_len=cfg.MAX_DEC_LEN + 10,
                dropout=d
            )
        else:
            self.dec_pos_enc = None

        dec_layer = nn.TransformerDecoderLayer(
            d_model=cfg.DEC_DIM,
            nhead=cfg.DEC_HEADS,
            dim_feedforward=cfg.DEC_FF,
            dropout=d,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.dec = nn.TransformerDecoder(dec_layer, num_layers=cfg.DEC_LAYERS)
        self.dec_ln = nn.LayerNorm(cfg.DEC_DIM)
        self.dec_head = nn.Linear(cfg.DEC_DIM, tok.dec_vocab)

        if cfg.USE_LM:
            self.lm_head = nn.Linear(cfg.DEC_DIM, tok.dec_vocab)

    def encode(self, imgs: torch.Tensor) -> torch.Tensor:
        x = self.stem(imgs)
        x = self.pos2d(x)
        x = F.adaptive_avg_pool2d(x, (1, x.size(-1)))
        x = x.squeeze(2).permute(0, 2, 1)
        x = self.enc_ln_in(x)
        x = self.enc(x)
        x = self.enc_ln(x)
        return x


# ========== PREPROCESSING ==========
class ResizeKeepRatioPadNoCrop:
    def __init__(self, h: int, w: int):
        self.h = h
        self.w = w

    def __call__(self, img: Image.Image) -> Image.Image:
        iw, ih = img.size
        if ih <= 0 or iw <= 0:
            return img.resize((self.w, self.h), Image.BILINEAR)

        scale = self.h / float(ih)
        nw = max(1, int(round(iw * scale)))
        img = img.resize((nw, self.h), Image.BILINEAR)

        if nw >= self.w:
            return img.crop((0, 0, self.w, self.h))

        # Left-aligned padding with gray (128)
        new_img = Image.new("L", (self.w, self.h), 128)
        new_img.paste(img, (0, 0))
        return new_img


def preprocess_pil(cfg: CFG, pil: Image.Image) -> torch.Tensor:
    img = pil.convert("L")
    img = ResizeKeepRatioPadNoCrop(cfg.IMG_H, cfg.IMG_W)(img)
    img_tensor = torch.from_numpy(np.array(img)).float() / 255.0
    img_tensor = (img_tensor - 0.5) / 0.5
    return img_tensor.unsqueeze(0).unsqueeze(0)


# ========== CONFIDENCE HELPERS ==========
def compute_ctc_confidence(
    ctc_logits: torch.Tensor, tok: CharTokenizer
) -> Tuple[float, str, int]:
    """
    Compute CTC confidence and decode text.

    Returns:
        (confidence, decoded_text, estimated_length)
    """
    if ctc_logits.dim() == 3:
        ctc_logits = ctc_logits.squeeze(0)  # [T, C]

    probs = F.softmax(ctc_logits, dim=-1)
    best_ids = ctc_logits.argmax(dim=-1).tolist()

    # Decode with deduplication
    text = tok.decode_ctc(best_ids)

    # Confidence: average max probability at each timestep
    max_probs = probs.max(dim=-1).values
    confidence = max_probs.mean().item()

    # Estimate length (count non-blank, non-repeated tokens)
    length = 0
    prev = None
    for idx in best_ids:
        if idx != prev and idx >= tok.ctc_offset:
            length += 1
        prev = idx

    return confidence, text, length


def compute_sequence_confidence(log_probs: List[float]) -> float:
    """Convert sequence log probabilities to confidence score."""
    if not log_probs:
        return 0.0

    # Average log prob
    avg_logp = sum(log_probs) / len(log_probs)

    # Convert to probability (clamped to [0, 1])
    confidence = math.exp(avg_logp)
    return min(1.0, max(0.0, confidence))


# ========== MAIN DECODING FUNCTION ==========
@torch.inference_mode()
def beam_decode_one_batched(
    model: KiriOCR,
    mem_proj_1: torch.Tensor,
    tok: CharTokenizer,
    cfg: CFG,
    ctc_logits_1: Optional[torch.Tensor] = None,
) -> Tuple[str, float]:
    """
    Beam search decoder with confidence score.

    Returns:
        (decoded_text, confidence_score)
    """
    device = mem_proj_1.device
    is_cuda = device.type == "cuda"

    # Get CTC info for length estimation and fusion
    ctc_confidence = None
    target_len = None

    if ctc_logits_1 is not None:
        ctc_confidence, ctc_text, target_len = compute_ctc_confidence(ctc_logits_1, tok)

    # Calculate max decoding steps
    if target_len and target_len > 0:
        max_steps = min(
            cfg.MAX_DEC_LEN,
            int(target_len * cfg.DEC_MAX_LEN_RATIO) + cfg.DEC_MAX_LEN_PAD,
        )
    else:
        mem_len = mem_proj_1.size(1)
        max_steps = min(
            cfg.MAX_DEC_LEN, int(mem_len * cfg.MEM_MAX_LEN_RATIO) + cfg.DEC_MAX_LEN_PAD
        )

    # Beam state: (score, token_ids, token_logprobs, finished)
    beams: List[Tuple[float, List[int], List[float], bool]] = [
        (0.0, [tok.dec_bos], [], False)
    ]

    full_causal = torch.triu(
        torch.ones(
            (cfg.MAX_DEC_LEN + 2, cfg.MAX_DEC_LEN + 2), device=device, dtype=torch.bool
        ),
        diagonal=1,
    )

    use_amp = cfg.USE_AUTOCAST and is_cuda

    for step in range(max_steps):
        if all(b[3] for b in beams):
            break

        alive = [b for b in beams if not b[3]]
        done = [b for b in beams if b[3]]

        if not alive:
            beams = done
            break

        maxL = max(len(b[1]) for b in alive)
        B = len(alive)

        inp = torch.full((B, maxL), tok.dec_pad, device=device, dtype=torch.long)
        for i, (_, seq, _, _) in enumerate(alive):
            inp[i, : len(seq)] = torch.tensor(seq, device=device, dtype=torch.long)

        tgt = model.dec_emb(inp)
        # Apply positional encoding if available (old models don't have it)
        if model.dec_pos_enc is not None:
            tgt = model.dec_pos_enc(tgt)
        causal = full_causal[:maxL, :maxL]

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            out = model.dec(
                tgt=tgt, memory=mem_proj_1.expand(B, -1, -1), tgt_mask=causal
            )
            out = model.dec_ln(out)
            logits = model.dec_head(out)[:, -1, :]
            logp = F.log_softmax(logits, dim=-1)

            if cfg.USE_LM and cfg.USE_LM_FUSION_EVAL and hasattr(model, "lm_head"):
                lm_logits = model.lm_head(out)[:, -1, :]
                logp = logp + cfg.LM_FUSION_ALPHA * F.log_softmax(lm_logits, dim=-1)

        # Apply penalties
        unk_id = tok.unk_id + tok.dec_offset

        for i, (_, seq, _, _) in enumerate(alive):
            cur_len = len(seq) - 1

            # EOS bias based on target length
            if target_len and target_len > 0:
                min_len = min(cfg.EOS_BIAS_UNTIL_LEN, max(1, int(target_len * 0.5)))
                if cur_len < min_len:
                    logp[i, tok.dec_eos] -= cfg.EOS_LOGP_BIAS
                elif cur_len >= target_len:
                    logp[i, tok.dec_eos] += cfg.EOS_LOGP_BOOST
            else:
                if cur_len < cfg.EOS_BIAS_UNTIL_LEN:
                    logp[i, tok.dec_eos] -= cfg.EOS_LOGP_BIAS

            # Repeat penalty - multiple patterns
            n = len(seq)
            
            # 1. Exact token repeat (e.g., AAA)
            if n >= 4 and seq[-1] == seq[-2] == seq[-3]:
                logp[i, seq[-1]] -= cfg.REPEAT_LAST_PENALTY
            
            # 2. Bi-gram repeat (e.g., AB-AB)
            if n >= 4:
                last_bigram = (seq[-2], seq[-1])
                prev_bigram = (seq[-4], seq[-3])
                if last_bigram == prev_bigram:
                    logp[i, seq[-1]] -= cfg.REPEAT_BIGRAM_PENALTY
                    logp[i, seq[-2]] -= cfg.REPEAT_BIGRAM_PENALTY
            
            # 3. Check for AB pattern starting to repeat (A-B-A)
            if n >= 3 and seq[-1] == seq[-3]:
                if n >= 4 and seq[-2] == seq[-4]:
                    logp[i, seq[-1]] -= cfg.REPEAT_BIGRAM_PENALTY
            
            # 4. Tri-gram repeat (e.g., ABC-ABC)
            if n >= 6:
                last_trigram = (seq[-3], seq[-2], seq[-1])
                prev_trigram = (seq[-6], seq[-5], seq[-4])
                if last_trigram == prev_trigram:
                    logp[i, seq[-1]] -= cfg.REPEAT_TRIGRAM_PENALTY
                    logp[i, seq[-2]] -= cfg.REPEAT_TRIGRAM_PENALTY
                    logp[i, seq[-3]] -= cfg.REPEAT_TRIGRAM_PENALTY

            # UNK penalty
            logp[i, unk_id] -= cfg.UNK_LOGP_PENALTY

        topv, topi = torch.topk(logp, k=cfg.BEAM, dim=-1)

        # Expand beams
        new_beams: List[Tuple[float, List[int], List[float], bool]] = list(done)

        for bi, (base_score, seq, logprobs, _) in enumerate(alive):
            for v, tid in zip(topv[bi].tolist(), topi[bi].tolist()):
                new_seq = seq + [int(tid)]
                new_logprobs = logprobs + [float(v)]
                is_finished = int(tid) == tok.dec_eos
                new_score = base_score + float(v)
                new_beams.append((new_score, new_seq, new_logprobs, is_finished))

        # Length-normalized scoring
        def normed(entry):
            score, seq, _, _ = entry
            L = max(1, len(seq) - 1)
            return score / (L**cfg.BEAM_LENP)

        new_beams.sort(key=normed, reverse=True)
        beams = new_beams[: cfg.BEAM]

    # ========== FINAL SCORING WITH CTC FUSION ==========
    def final_score_and_confidence(entry):
        score, seq, logprobs, _ = entry
        length = max(1, len(seq) - 1)

        dec_score = score / (length**cfg.BEAM_LENP)
        dec_conf = compute_sequence_confidence(logprobs)

        # CTC fusion
        if ctc_logits_1 is not None and cfg.CTC_FUSION_ALPHA > 0:
            ctc_score = compute_ctc_alignment_score(ctc_logits_1, seq, tok)
            combined_score = dec_score + cfg.CTC_FUSION_ALPHA * ctc_score
            return combined_score, dec_conf

        return dec_score, dec_conf

    scored = [(final_score_and_confidence(b), b) for b in beams]
    scored.sort(key=lambda x: x[0][0], reverse=True)

    (_, best_dec_conf), best_beam = scored[0]
    _, best_seq, _, _ = best_beam

    # Decode text
    ids = []
    for x in best_seq[1:]:
        if x == tok.dec_eos:
            break
        ids.append(x)
    text = tok.decode_dec(ids)

    # ========== COMPUTE FINAL CONFIDENCE ==========
    # Combine decoder confidence with CTC confidence (if available)
    if ctc_confidence is not None:
        # Weighted average: favor decoder slightly
        final_confidence = 0.6 * best_dec_conf + 0.4 * ctc_confidence
    else:
        final_confidence = best_dec_conf

    return text, final_confidence


def compute_ctc_alignment_score(
    ctc_logits: torch.Tensor, dec_seq: List[int], tok: CharTokenizer
) -> float:
    """Compute CTC alignment score using forward algorithm."""
    if ctc_logits.dim() == 3:
        ctc_logits = ctc_logits.squeeze(0)

    log_probs = F.log_softmax(ctc_logits, dim=-1)

    # Convert decoder sequence to CTC labels
    labels = []
    for x in dec_seq[1:]:
        if x == tok.dec_eos:
            break
        if x in (tok.dec_pad, tok.dec_bos):
            continue
        ctc_id = tok.dec_to_ctc_id(x)
        labels.append(ctc_id)

    if not labels:
        return log_probs[:, tok.blank_id].sum().item() / max(1, log_probs.size(0))

    # Forward algorithm
    T = log_probs.size(0)
    blank = tok.blank_id

    # Extend: [b, l0, b, l1, b, ...]
    ext = [blank]
    for lid in labels:
        ext.append(lid)
        ext.append(blank)
    S = len(ext)

    alpha = log_probs.new_full((S,), float("-inf"))
    alpha[0] = log_probs[0, blank]
    if S > 1:
        alpha[1] = log_probs[0, ext[1]]

    for t in range(1, T):
        new_alpha = log_probs.new_full((S,), float("-inf"))
        for s in range(S):
            candidates = [alpha[s]]
            if s > 0:
                candidates.append(alpha[s - 1])
            if s > 1 and ext[s] != blank and ext[s] != ext[s - 2]:
                candidates.append(alpha[s - 2])

            stacked = torch.stack(
                [
                    (
                        c
                        if isinstance(c, torch.Tensor)
                        else torch.tensor(c, device=log_probs.device)
                    )
                    for c in candidates
                ]
            )
            new_alpha[s] = torch.logsumexp(stacked, dim=0) + log_probs[t, ext[s]]
        alpha = new_alpha

    if S == 1:
        total = alpha[0]
    else:
        total = torch.logsumexp(torch.stack([alpha[S - 1], alpha[S - 2]]), dim=0)

    return total.item() / max(1, len(labels))


# ========== GREEDY CTC DECODE (FAST ALTERNATIVE) ==========
@torch.inference_mode()
def greedy_ctc_decode(
    model: KiriOCR, imgs: torch.Tensor, tok: CharTokenizer, cfg: CFG
) -> Tuple[str, float]:
    """
    Fast greedy CTC decoding (no beam search).

    Returns:
        (text, confidence)
    """
    mem = model.encode(imgs[:1])
    ctc_logits = model.ctc_head(mem)

    confidence, text, _ = compute_ctc_confidence(ctc_logits, tok)
    return text, confidence


@torch.inference_mode()
def greedy_ctc_decode_streaming(
    model: KiriOCR,
    mem: torch.Tensor,
    tok: CharTokenizer,
    cfg: CFG,
) -> Generator[Dict, None, None]:
    """
    CTC greedy decoding with frame-by-frame streaming output.
    
    Unlike decoder-based streaming, CTC decodes all frames at once but
    we simulate streaming by yielding characters as they appear during
    the deduplication process.
    
    Args:
        model: The KiriOCR model
        mem: Encoder memory [1, T, D] (NOT projected - need raw encoder output)
        tok: Character tokenizer
        cfg: Model configuration
        
    Yields:
        Dict with:
        - 'token': The new character/token string
        - 'token_id': The CTC token ID
        - 'text': Full decoded text so far
        - 'confidence': Frame probability
        - 'step': Current frame number
        - 'finished': Whether decoding is complete
    """
    # Get CTC logits
    ctc_logits = model.ctc_head(mem)
    
    if ctc_logits.dim() == 3:
        ctc_logits = ctc_logits.squeeze(0)  # [T, C]
    
    probs = F.softmax(ctc_logits, dim=-1)
    T = ctc_logits.size(0)
    
    # Track decoded text and previous token for deduplication
    decoded_text = ""
    prev_id = None
    step = 0
    
    # Get all best IDs at once
    best_ids = ctc_logits.argmax(dim=-1)  # [T]
    max_probs = probs.max(dim=-1).values  # [T]
    
    for t in range(T):
        idx = best_ids[t].item()
        conf = max_probs[t].item()
        
        # CTC deduplication: skip if same as previous or blank
        if idx == prev_id:
            continue
        
        prev_id = idx
        
        # Skip blank and special tokens
        if idx < tok.ctc_offset:
            continue
            
        # Decode character
        raw_id = idx - tok.ctc_offset
        if 0 <= raw_id < tok.vocab_size:
            char = tok.id_to_token.get(raw_id, "")
            if char and char != tok.unk_token:
                decoded_text += char
                step += 1
                
                yield {
                    "token": char,
                    "token_id": idx,
                    "text": decoded_text,
                    "confidence": conf,
                    "step": step,
                    "finished": False,
                }
    
    # Final yield to signal completion
    yield {
        "token": "",
        "token_id": -1,
        "text": decoded_text,
        "confidence": probs.max(dim=-1).values.mean().item(),
        "step": step,
        "finished": True,
    }


# ========== STREAMING DECODER (CHARACTER-BY-CHARACTER) ==========
@torch.inference_mode()
def greedy_decode_streaming(
    model: KiriOCR,
    mem_proj_1: torch.Tensor,
    tok: CharTokenizer,
    cfg: CFG,
    ctc_logits_1: Optional[torch.Tensor] = None,
) -> Generator[Dict, None, None]:
    """
    Greedy autoregressive decoder that yields each character as it's generated.
    
    Similar to LLM text streaming - yields one token at a time during inference.
    Uses greedy decoding (beam_size=1) for real-time streaming.
    
    Args:
        model: The KiriOCR model
        mem_proj_1: Projected encoder memory [1, T, D]
        tok: Character tokenizer
        cfg: Model configuration
        ctc_logits_1: Optional CTC logits for length estimation
        
    Yields:
        Dict with:
        - 'token': The new character/token string
        - 'token_id': The token ID
        - 'text': Full decoded text so far
        - 'confidence': Token probability
        - 'step': Current step number
        - 'finished': Whether decoding is complete
    """
    device = mem_proj_1.device
    is_cuda = device.type == "cuda"
    use_amp = cfg.USE_AUTOCAST and is_cuda
    
    # Estimate target length from CTC if available
    target_len = None
    if ctc_logits_1 is not None:
        _, _, target_len = compute_ctc_confidence(ctc_logits_1, tok)
    
    # Calculate max decoding steps
    if target_len and target_len > 0:
        max_steps = min(
            cfg.MAX_DEC_LEN,
            int(target_len * cfg.DEC_MAX_LEN_RATIO) + cfg.DEC_MAX_LEN_PAD,
        )
    else:
        mem_len = mem_proj_1.size(1)
        max_steps = min(
            cfg.MAX_DEC_LEN, int(mem_len * cfg.MEM_MAX_LEN_RATIO) + cfg.DEC_MAX_LEN_PAD
        )
    
    # Initialize with BOS token
    generated_ids = [tok.dec_bos]
    generated_text = ""
    log_probs = []
    
    # Pre-compute causal mask
    full_causal = torch.triu(
        torch.ones(
            (cfg.MAX_DEC_LEN + 2, cfg.MAX_DEC_LEN + 2), device=device, dtype=torch.bool
        ),
        diagonal=1,
    )
    
    unk_id = tok.unk_id + tok.dec_offset
    
    for step in range(max_steps):
        # Prepare input sequence
        seq_len = len(generated_ids)
        inp = torch.tensor([generated_ids], device=device, dtype=torch.long)
        
        tgt = model.dec_emb(inp)
        # Apply positional encoding if available (old models don't have it)
        if model.dec_pos_enc is not None:
            tgt = model.dec_pos_enc(tgt)
        causal = full_causal[:seq_len, :seq_len]
        
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            out = model.dec(tgt=tgt, memory=mem_proj_1, tgt_mask=causal)
            out = model.dec_ln(out)
            logits = model.dec_head(out)[:, -1, :]  # [1, vocab]
            logp = F.log_softmax(logits, dim=-1)
            
            # Apply LM fusion if available
            if cfg.USE_LM and cfg.USE_LM_FUSION_EVAL and hasattr(model, "lm_head"):
                lm_logits = model.lm_head(out)[:, -1, :]
                logp = logp + cfg.LM_FUSION_ALPHA * F.log_softmax(lm_logits, dim=-1)
        
        cur_len = len(generated_ids) - 1
        
        # EOS bias based on target length
        if target_len and target_len > 0:
            min_len = min(cfg.EOS_BIAS_UNTIL_LEN, max(1, int(target_len * 0.5)))
            if cur_len < min_len:
                logp[0, tok.dec_eos] -= cfg.EOS_LOGP_BIAS
            elif cur_len >= target_len:
                logp[0, tok.dec_eos] += cfg.EOS_LOGP_BOOST
        else:
            if cur_len < cfg.EOS_BIAS_UNTIL_LEN:
                logp[0, tok.dec_eos] -= cfg.EOS_LOGP_BIAS
        
        # Repeat penalty - multiple patterns
        n = len(generated_ids)
        
        # 1. Exact token repeat (e.g., AAA)
        if n >= 4 and generated_ids[-1] == generated_ids[-2] == generated_ids[-3]:
            logp[0, generated_ids[-1]] -= cfg.REPEAT_LAST_PENALTY
        
        # 2. Bi-gram repeat (e.g., AB-AB -> penalize next A or B)
        if n >= 4:
            last_bigram = (generated_ids[-2], generated_ids[-1])
            prev_bigram = (generated_ids[-4], generated_ids[-3])
            if last_bigram == prev_bigram:
                # Penalize both tokens of the bigram
                logp[0, generated_ids[-1]] -= cfg.REPEAT_BIGRAM_PENALTY
                logp[0, generated_ids[-2]] -= cfg.REPEAT_BIGRAM_PENALTY
        
        # 3. Check for AB pattern starting to repeat (A-B-A)
        if n >= 3 and generated_ids[-1] == generated_ids[-3]:
            # Current might be forming AB-AB pattern, penalize repeating the cycle
            if n >= 4 and generated_ids[-2] == generated_ids[-4]:
                logp[0, generated_ids[-1]] -= cfg.REPEAT_BIGRAM_PENALTY
        
        # 4. Tri-gram repeat (e.g., ABC-ABC)
        if n >= 6:
            last_trigram = (generated_ids[-3], generated_ids[-2], generated_ids[-1])
            prev_trigram = (generated_ids[-6], generated_ids[-5], generated_ids[-4])
            if last_trigram == prev_trigram:
                logp[0, generated_ids[-1]] -= cfg.REPEAT_TRIGRAM_PENALTY
                logp[0, generated_ids[-2]] -= cfg.REPEAT_TRIGRAM_PENALTY
                logp[0, generated_ids[-3]] -= cfg.REPEAT_TRIGRAM_PENALTY
        
        # UNK penalty
        logp[0, unk_id] -= cfg.UNK_LOGP_PENALTY
        
        # Get best token (greedy)
        probs = F.softmax(logits, dim=-1)
        best_prob, best_id = probs[0].max(dim=0)
        best_id = best_id.item()
        best_logp = logp[0, best_id].item()
        
        # Check if finished
        is_finished = best_id == tok.dec_eos
        
        # Decode character
        if not is_finished and best_id not in (tok.dec_pad, tok.dec_bos, tok.dec_eos):
            raw_id = best_id - tok.dec_offset
            if 0 <= raw_id < tok.vocab_size:
                char = tok.id_to_token.get(raw_id, "")
                if char != tok.unk_token:
                    generated_text += char
        else:
            char = ""
        
        generated_ids.append(best_id)
        log_probs.append(best_logp)
        
        yield {
            "token": char,
            "token_id": best_id,
            "text": generated_text,
            "confidence": best_prob.item(),
            "step": step + 1,
            "finished": is_finished,
        }
        
        if is_finished:
            break


@torch.inference_mode()
def beam_decode_streaming(
    model: KiriOCR,
    mem_proj_1: torch.Tensor,
    tok: CharTokenizer,
    cfg: CFG,
    ctc_logits_1: Optional[torch.Tensor] = None,
) -> Generator[Dict, None, None]:
    """
    Beam search decoder that yields the current best hypothesis at each step.
    
    Provides higher quality than greedy decoding while still streaming.
    At each step, yields the best partial result from the beam.
    
    Args:
        model: The KiriOCR model
        mem_proj_1: Projected encoder memory [1, T, D]
        tok: Character tokenizer
        cfg: Model configuration
        ctc_logits_1: Optional CTC logits for length estimation
        
    Yields:
        Dict with:
        - 'token': The new character (may change in later steps due to beam search!)
        - 'text': Current best decoded text
        - 'confidence': Current confidence estimate
        - 'step': Current step number
        - 'finished': Whether decoding is complete
    """
    device = mem_proj_1.device
    is_cuda = device.type == "cuda"
    
    # Get CTC info for length estimation
    ctc_confidence = None
    target_len = None
    
    if ctc_logits_1 is not None:
        ctc_confidence, _, target_len = compute_ctc_confidence(ctc_logits_1, tok)
    
    # Calculate max decoding steps
    if target_len and target_len > 0:
        max_steps = min(
            cfg.MAX_DEC_LEN,
            int(target_len * cfg.DEC_MAX_LEN_RATIO) + cfg.DEC_MAX_LEN_PAD,
        )
    else:
        mem_len = mem_proj_1.size(1)
        max_steps = min(
            cfg.MAX_DEC_LEN, int(mem_len * cfg.MEM_MAX_LEN_RATIO) + cfg.DEC_MAX_LEN_PAD
        )
    
    # Beam state: (score, token_ids, token_logprobs, finished)
    beams: List[Tuple[float, List[int], List[float], bool]] = [
        (0.0, [tok.dec_bos], [], False)
    ]
    
    full_causal = torch.triu(
        torch.ones(
            (cfg.MAX_DEC_LEN + 2, cfg.MAX_DEC_LEN + 2), device=device, dtype=torch.bool
        ),
        diagonal=1,
    )
    
    use_amp = cfg.USE_AUTOCAST and is_cuda
    prev_best_text = ""
    
    for step in range(max_steps):
        if all(b[3] for b in beams):
            break
        
        alive = [b for b in beams if not b[3]]
        done = [b for b in beams if b[3]]
        
        if not alive:
            beams = done
            break
        
        maxL = max(len(b[1]) for b in alive)
        B = len(alive)
        
        inp = torch.full((B, maxL), tok.dec_pad, device=device, dtype=torch.long)
        for i, (_, seq, _, _) in enumerate(alive):
            inp[i, : len(seq)] = torch.tensor(seq, device=device, dtype=torch.long)
        
        tgt = model.dec_emb(inp)
        # Apply positional encoding if available (old models don't have it)
        if model.dec_pos_enc is not None:
            tgt = model.dec_pos_enc(tgt)
        causal = full_causal[:maxL, :maxL]
        
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            out = model.dec(
                tgt=tgt, memory=mem_proj_1.expand(B, -1, -1), tgt_mask=causal
            )
            out = model.dec_ln(out)
            logits = model.dec_head(out)[:, -1, :]
            logp = F.log_softmax(logits, dim=-1)
            
            if cfg.USE_LM and cfg.USE_LM_FUSION_EVAL and hasattr(model, "lm_head"):
                lm_logits = model.lm_head(out)[:, -1, :]
                logp = logp + cfg.LM_FUSION_ALPHA * F.log_softmax(lm_logits, dim=-1)
        
        # Apply penalties
        unk_id = tok.unk_id + tok.dec_offset
        
        for i, (_, seq, _, _) in enumerate(alive):
            cur_len = len(seq) - 1
            
            if target_len and target_len > 0:
                min_len = min(cfg.EOS_BIAS_UNTIL_LEN, max(1, int(target_len * 0.5)))
                if cur_len < min_len:
                    logp[i, tok.dec_eos] -= cfg.EOS_LOGP_BIAS
                elif cur_len >= target_len:
                    logp[i, tok.dec_eos] += cfg.EOS_LOGP_BOOST
            else:
                if cur_len < cfg.EOS_BIAS_UNTIL_LEN:
                    logp[i, tok.dec_eos] -= cfg.EOS_LOGP_BIAS
            
            # Repeat penalty - multiple patterns
            n = len(seq)
            
            # 1. Exact token repeat (e.g., AAA)
            if n >= 4 and seq[-1] == seq[-2] == seq[-3]:
                logp[i, seq[-1]] -= cfg.REPEAT_LAST_PENALTY
            
            # 2. Bi-gram repeat (e.g., AB-AB)
            if n >= 4:
                last_bigram = (seq[-2], seq[-1])
                prev_bigram = (seq[-4], seq[-3])
                if last_bigram == prev_bigram:
                    logp[i, seq[-1]] -= cfg.REPEAT_BIGRAM_PENALTY
                    logp[i, seq[-2]] -= cfg.REPEAT_BIGRAM_PENALTY
            
            # 3. Check for AB pattern starting to repeat (A-B-A)
            if n >= 3 and seq[-1] == seq[-3]:
                if n >= 4 and seq[-2] == seq[-4]:
                    logp[i, seq[-1]] -= cfg.REPEAT_BIGRAM_PENALTY
            
            # 4. Tri-gram repeat (e.g., ABC-ABC)
            if n >= 6:
                last_trigram = (seq[-3], seq[-2], seq[-1])
                prev_trigram = (seq[-6], seq[-5], seq[-4])
                if last_trigram == prev_trigram:
                    logp[i, seq[-1]] -= cfg.REPEAT_TRIGRAM_PENALTY
                    logp[i, seq[-2]] -= cfg.REPEAT_TRIGRAM_PENALTY
                    logp[i, seq[-3]] -= cfg.REPEAT_TRIGRAM_PENALTY
            
            logp[i, unk_id] -= cfg.UNK_LOGP_PENALTY
        
        topv, topi = torch.topk(logp, k=cfg.BEAM, dim=-1)
        
        # Expand beams
        new_beams: List[Tuple[float, List[int], List[float], bool]] = list(done)
        
        for bi, (base_score, seq, logprobs, _) in enumerate(alive):
            for v, tid in zip(topv[bi].tolist(), topi[bi].tolist()):
                new_seq = seq + [int(tid)]
                new_logprobs = logprobs + [float(v)]
                is_finished = int(tid) == tok.dec_eos
                new_score = base_score + float(v)
                new_beams.append((new_score, new_seq, new_logprobs, is_finished))
        
        # Length-normalized scoring
        def normed(entry):
            score, seq, _, _ = entry
            L = max(1, len(seq) - 1)
            return score / (L ** cfg.BEAM_LENP)
        
        new_beams.sort(key=normed, reverse=True)
        beams = new_beams[: cfg.BEAM]
        
        # Get current best beam for streaming output
        best_beam = beams[0]
        _, best_seq, best_logprobs, best_finished = best_beam
        
        # Decode current best text
        ids = []
        for x in best_seq[1:]:
            if x == tok.dec_eos:
                break
            ids.append(x)
        current_text = tok.decode_dec(ids)
        
        # Find new token
        if len(current_text) > len(prev_best_text):
            new_token = current_text[len(prev_best_text):]
        else:
            new_token = ""
        
        # Compute confidence
        confidence = compute_sequence_confidence(best_logprobs) if best_logprobs else 0.0
        
        yield {
            "token": new_token,
            "text": current_text,
            "confidence": confidence,
            "step": step + 1,
            "finished": best_finished,
        }
        
        prev_best_text = current_text
        
        if best_finished:
            break
