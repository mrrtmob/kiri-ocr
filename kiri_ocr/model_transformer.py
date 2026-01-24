# kiri_ocr/model_transformer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import json
import math
import numpy as np
from torchvision import transforms
from PIL import Image, ImageOps

class CFG:
    # --- Model Architecture ---
    IMG_H: int = 48
    IMG_W: int = 640
    MAX_DEC_LEN: int = 260
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

    # --- Inference Params (STRICT MODE) ---
    
    # 1. High CTC Fusion
    # Forces the decoder to respect the visual boundaries found by CTC.
    # CTC usually outputs blanks after the word, so this kills the repetition.
    CTC_FUSION_ALPHA: float = 0.8  

    # 2. Narrow Beam
    # Reducing from 6 to 2 prevents the model from "overthinking" 
    # and finding weird long paths like "CONTAIN CON".
    BEAM: int = 2
    
    # 3. No Length Bonus
    # Keep at 0.0. Any positive number rewards longer text (like adding " CON").
    BEAM_LENP: float = 0.0

    # 4. Aggressive Stopping
    # Massive bias to force the End-Of-Sentence token.
    EOS_LOGP_BIAS: float = 10.0
    EOS_LOGP_BOOST: float = 10.0
    EOS_BIAS_UNTIL_LEN: int = 1

    # 5. Anti-Loop
    REPEAT_LAST_PENALTY: float = 5.0

    # Misc
    DEC_MAX_LEN_RATIO: float = 1.2
    DEC_MAX_LEN_PAD: int = 6
    MEM_MAX_LEN_RATIO: float = 0.75
    UNK_LOGP_PENALTY: float = 1.0

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

class PosEnc2D(nn.Module):
    """Dynamic 2D Positional Encoding - matches working inference code"""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def _make_pe(self, length: int, dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        pos = torch.arange(length, dtype=dtype, device=device).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2, dtype=dtype, device=device) * (-math.log(10000.0) / dim))
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
            nn.Conv2d(1, 48, 3, 1, 1, bias=False), nn.BatchNorm2d(48), nn.SiLU(inplace=True),
            nn.Conv2d(48, 96, 3, (2, 2), 1, bias=False), nn.BatchNorm2d(96), nn.SiLU(inplace=True),
            nn.Conv2d(96, 160, 3, (2, 2), 1, bias=False), nn.BatchNorm2d(160), nn.SiLU(inplace=True),
            nn.Conv2d(160, dim, 3, (2, 1), 1, bias=False), nn.BatchNorm2d(dim), nn.SiLU(inplace=True),
            nn.Dropout2d(dropout),
        )
    def forward(self, x): return self.net(x)

class HybridContextOCRV2(nn.Module):
    def __init__(self, cfg: CFG, tok: CharTokenizer):
        super().__init__()
        self.cfg = cfg
        d = cfg.DROPOUT
        self.stem = ConvStem(cfg.ENC_DIM, d)
        
        # Use the working PosEnc2D implementation
        self.pos2d = PosEnc2D(cfg.ENC_DIM)
        
        self.enc_ln_in = nn.LayerNorm(cfg.ENC_DIM)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.ENC_DIM, 
            nhead=cfg.ENC_HEADS, 
            dim_feedforward=cfg.ENC_FF, 
            dropout=d, 
            batch_first=True, 
            activation="gelu", 
            norm_first=True
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=cfg.ENC_LAYERS)
        self.enc_ln = nn.LayerNorm(cfg.ENC_DIM)
        
        if cfg.USE_CTC:
            self.ctc_head = nn.Sequential(
                nn.LayerNorm(cfg.ENC_DIM), 
                nn.Dropout(d), 
                nn.Linear(cfg.ENC_DIM, tok.ctc_classes)
            )
        
        self.mem_proj = nn.Linear(cfg.ENC_DIM, cfg.DEC_DIM, bias=False)
        self.dec_emb = nn.Embedding(tok.dec_vocab, cfg.DEC_DIM)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=cfg.DEC_DIM, 
            nhead=cfg.DEC_HEADS, 
            dim_feedforward=cfg.DEC_FF, 
            dropout=d, 
            batch_first=True, 
            activation="gelu", 
            norm_first=True
        )
        self.dec = nn.TransformerDecoder(dec_layer, num_layers=cfg.DEC_LAYERS)
        self.dec_ln = nn.LayerNorm(cfg.DEC_DIM)
        self.dec_head = nn.Linear(cfg.DEC_DIM, tok.dec_vocab)
        self.lm_head = nn.Linear(cfg.DEC_DIM, tok.dec_vocab)

    def encode(self, imgs: torch.Tensor) -> torch.Tensor:
        x = self.stem(imgs)
        x = self.pos2d(x)
        
        # CRITICAL: Pool height to 1 before flattening (1D sequence)
        x = F.adaptive_avg_pool2d(x, (1, x.size(-1)))
        
        x = x.squeeze(2).permute(0, 2, 1)  # [B, W', C]
        x = self.enc_ln_in(x)
        x = self.enc(x)
        x = self.enc_ln(x)
        return x

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

        if nw == self.w:
            return img
        if nw < self.w:
            # Pad right (or center?) - Training used left-aligned (paste at 0,0)
            # app.py used center padding: ImageOps.expand(..., fill=255)
            # We should match TRAINING which used paste((0,0)).
            # app.py logic:
            #   pad_total = self.w - nw
            #   left = pad_total // 2
            #   right = pad_total - left
            #   ImageOps.expand...
            
            # Let's check training_transformer.py again.
            # final_img.paste(img, (0, 0)) -> Left aligned.
            # So app.py is slightly different (Center aligned).
            # If the model was trained with left alignment, we should use left alignment.
            # However, the user provided app.py as "example infer", implying it works.
            # Let's stick to left alignment if training code did that,
            # OR provide a flag.
            
            # Re-reading training code:
            # final_img = Image.new("L", (self.img_width, self.img_height), 128)
            # final_img.paste(img.crop((0,0, paste_w, self.img_height)), (0, 0))
            # It is LEFT aligned with 128 gray padding.
            
            # The user's app.py uses ImageOps.expand with fill=255 (white).
            # This is a discrepancy.
            # If I use the user's app.py logic, it might be what they want.
            # But technically training used 128 padding and left align.
            # I will implement Left Align + 128 padding to match training,
            # unless user strictly wants app.py behavior.
            # But the user said "show to use that lib like my ocr class".
            # I will use the app.py logic for now but maybe comment on it.
            # Wait, app.py uses `ResizeKeepRatioPadNoCrop`.
            
            new_img = Image.new("L", (self.w, self.h), 128)
            new_img.paste(img, (0, 0))
            return new_img

        return img.resize((self.w, self.h), Image.BILINEAR)

def preprocess_pil(cfg: CFG, pil: Image.Image) -> torch.Tensor:
    img = pil.convert("RGB")
    # Convert to grayscale for model
    img = img.convert("L")
    
    img = ResizeKeepRatioPadNoCrop(cfg.IMG_H, cfg.IMG_W)(img)
    
    # Normalize
    img_tensor = torch.from_numpy(np.array(img)).float() / 255.0
    img_tensor = (img_tensor - 0.5) / 0.5
    
    return img_tensor.unsqueeze(0).unsqueeze(0) # [1, 1, H, W]

@torch.inference_mode()
def estimate_ctc_length(ctc_logits_1: torch.Tensor, tok: CharTokenizer) -> int:
    x = ctc_logits_1[0] if ctc_logits_1.dim() == 3 else ctc_logits_1
    ids = x.argmax(dim=-1).tolist()
    prev = None
    length = 0
    for i in ids:
        if i == prev:
            continue
        prev = i
        if i <= tok.pad_id:
            continue
        length += 1
    return length

@torch.inference_mode()
def beam_decode_one_batched(
    model: HybridContextOCRV2,
    mem_proj_1: torch.Tensor,
    tok: CharTokenizer,
    cfg: CFG,
    ctc_logits_1: Optional[torch.Tensor] = None,
) -> str:
    device = mem_proj_1.device
    is_cuda = device.type == "cuda"

    beams: List[Tuple[float, List[int], bool]] = [(0.0, [tok.dec_bos], False)]

    max_steps = cfg.MAX_DEC_LEN
    target_len = None
    if ctc_logits_1 is not None:
        target_len = estimate_ctc_length(ctc_logits_1, tok)
        if target_len > 0:
            max_steps = min(max_steps, max(1, int(target_len * cfg.DEC_MAX_LEN_RATIO) + cfg.DEC_MAX_LEN_PAD))
        else:
            max_steps = min(max_steps, cfg.DEC_MAX_LEN_PAD)
    else:
        mem_len = mem_proj_1.size(1)
        max_steps = min(max_steps, max(1, int(mem_len * cfg.MEM_MAX_LEN_RATIO) + cfg.DEC_MAX_LEN_PAD))

    full_causal = torch.triu(
        torch.ones((cfg.MAX_DEC_LEN + 2, cfg.MAX_DEC_LEN + 2), device=device, dtype=torch.bool),
        diagonal=1
    )

    use_amp = cfg.USE_AUTOCAST and is_cuda

    for _ in range(max_steps):
        if all(b[2] for b in beams):
            break

        alive = [b for b in beams if not b[2]]
        done = [b for b in beams if b[2]]

        if len(alive) == 0:
            beams = done
            break

        maxL = max(len(b[1]) for b in alive)
        B = len(alive)

        inp = torch.full((B, maxL), tok.dec_pad, device=device, dtype=torch.long)
        for i, (_, seq, _) in enumerate(alive):
            inp[i, :len(seq)] = torch.tensor(seq, device=device, dtype=torch.long)

        tgt = model.dec_emb(inp)
        causal = full_causal[:maxL, :maxL]

        # Handle autocast manually if needed, or rely on context
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            out = model.dec(tgt=tgt, memory=mem_proj_1.expand(B, -1, -1), tgt_mask=causal)
            out = model.dec_ln(out)
            logits = model.dec_head(out)[:, -1, :]
            logp = F.log_softmax(logits, dim=-1)

            if cfg.USE_LM and cfg.USE_LM_FUSION_EVAL:
                lm_logits = model.lm_head(out)[:, -1, :]
                logp = logp + cfg.LM_FUSION_ALPHA * F.log_softmax(lm_logits, dim=-1)

        unk_id = tok.unk_id + tok.dec_offset

        for i, (_, seq, _) in enumerate(alive):
            cur_len = len(seq) - 1

            if target_len is not None and target_len > 0:
                min_len = min(cfg.EOS_BIAS_UNTIL_LEN, max(1, int(target_len * 0.7)))
                if cur_len < min_len:
                    logp[i, tok.dec_eos] = logp[i, tok.dec_eos] - cfg.EOS_LOGP_BIAS
                elif cur_len >= target_len:
                    logp[i, tok.dec_eos] = logp[i, tok.dec_eos] + cfg.EOS_LOGP_BOOST
            else:
                if cur_len < cfg.EOS_BIAS_UNTIL_LEN:
                    logp[i, tok.dec_eos] = logp[i, tok.dec_eos] - cfg.EOS_LOGP_BIAS

            if len(seq) >= 4 and seq[-1] == seq[-2] == seq[-3]:
                logp[i, seq[-1]] = logp[i, seq[-1]] - cfg.REPEAT_LAST_PENALTY

            logp[i, unk_id] = logp[i, unk_id] - cfg.UNK_LOGP_PENALTY

        topv, topi = torch.topk(logp, k=cfg.BEAM, dim=-1)

        new_beams: List[Tuple[float, List[int], bool]] = []
        new_beams.extend(done)

        for bi, (base_score, seq, _) in enumerate(alive):
            for v, tid in zip(topv[bi].tolist(), topi[bi].tolist()):
                ns = seq + [int(tid)]
                nf = (int(tid) == tok.dec_eos)
                new_beams.append((base_score + float(v), ns, nf))

        def normed(s: float, seq_: List[int]) -> float:
            L2 = max(1, len(seq_) - 1)
            return s / (L2 ** cfg.BEAM_LENP)

        new_beams.sort(key=lambda x: normed(x[0], x[1]), reverse=True)
        beams = new_beams[:cfg.BEAM]

    def length_norm(score: float, seq: List[int]) -> float:
        return score / (max(1, len(seq) - 1) ** cfg.BEAM_LENP)

    if ctc_logits_1 is not None and cfg.CTC_FUSION_ALPHA > 0:
        log_probs = F.log_softmax(ctc_logits_1.squeeze(0), dim=-1)

        def ctc_sequence_log_prob(label_ids: List[int]) -> torch.Tensor:
            if len(label_ids) == 0:
                return log_probs[:, tok.blank_id].sum()

            blank = tok.blank_id
            ext = [blank]
            for lid in label_ids:
                ext.append(lid)
                ext.append(blank)

            s_len = len(ext)
            alpha = log_probs.new_full((s_len,), float("-inf"))
            alpha[0] = log_probs[0, blank]
            alpha[1] = log_probs[0, ext[1]]

            for t in range(1, log_probs.size(0)):
                next_alpha = log_probs.new_full((s_len,), float("-inf"))
                for s in range(s_len):
                    candidates = [alpha[s]]
                    if s - 1 >= 0:
                        candidates.append(alpha[s - 1])
                    if s - 2 >= 0 and ext[s] != blank and ext[s] != ext[s - 2]:
                        candidates.append(alpha[s - 2])
                    next_alpha[s] = torch.logsumexp(torch.stack(candidates), dim=0) + log_probs[t, ext[s]]
                alpha = next_alpha

            if s_len == 1:
                return alpha[0]
            return torch.logsumexp(torch.stack([alpha[s_len - 1], alpha[s_len - 2]]), dim=0)

        def seq_to_ctc_labels(seq: List[int]) -> List[int]:
            labels = []
            for x in seq[1:]:
                if x == tok.dec_eos:
                    break
                if x in (tok.dec_pad, tok.dec_bos):
                    continue
                y = x - tok.dec_offset
                if 0 <= y < tok.vocab_size:
                    labels.append(y + tok.ctc_offset)
                else:
                    labels.append(tok.unk_id + tok.ctc_offset)
            return labels

        def combined_score(entry):
            dec_score = length_norm(entry[0], entry[1])
            labels = seq_to_ctc_labels(entry[1])
            ctc_score = ctc_sequence_log_prob(labels) / max(1, len(labels))
            return dec_score + cfg.CTC_FUSION_ALPHA * float(ctc_score)

        best = max(beams, key=combined_score)[1]
    else:
        best = max(beams, key=lambda x: length_norm(x[0], x[1]))[1]

    ids = []
    for x in best[1:]:
        if x == tok.dec_eos:
            break
        ids.append(x)
    return tok.decode_dec(ids)