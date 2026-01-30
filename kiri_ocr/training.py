"""
KiriOCR Transformer Training Module.

This module contains training code for the main Transformer OCR model
using hybrid CTC + attention decoder loss.
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import OneCycleLR
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import numpy as np
import json
import math
import itertools

try:
    from datasets import load_dataset, concatenate_datasets
except ImportError:
    load_dataset = None
    concatenate_datasets = None

try:
    from safetensors.torch import save_file, load_file
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False
    print("⚠️ safetensors not installed. Using torch.save instead.")

from .model import KiriOCR, CFG, CharTokenizer, greedy_ctc_decode


# ========== VOCAB BUILDERS ==========
def build_vocab_from_hf_dataset(dataset, output_path, text_col="text"):
    """Scans a HF dataset to create vocab_char.json automatically."""
    print(f"📖 Scanning HF Dataset to build vocabulary...")
    unique_chars = set()

    try:
        iter_ds = dataset.select_columns([text_col])
    except:
        iter_ds = dataset

    for item in tqdm(iter_ds, desc="Scanning Vocab"):
        text = item.get(text_col, "")
        if text:
            unique_chars.update(list(text))

    print(f"   Found {len(unique_chars)} unique characters.")

    sorted_chars = sorted(list(unique_chars))
    vocab = {"<unk>": 0}
    idx = 1
    for char in sorted_chars:
        if char != "<unk>":
            vocab[char] = idx
            idx += 1

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=2, ensure_ascii=False)

    print(f"✅ Generated vocabulary saved to: {output_path}")
    return output_path


def build_vocab_from_dataset(labels_file, output_path):
    """Scans the training file and creates a vocab_char.json automatically."""
    print(f"📖 Scanning {labels_file} to build vocabulary...")
    unique_chars = set()

    try:
        with open(labels_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    text = parts[1]
                    unique_chars.update(list(text))
    except Exception as e:
        print(f"❌ Error reading labels file: {e}")
        return None

    print(f"   Found {len(unique_chars)} unique characters.")
    sorted_chars = sorted(list(unique_chars))

    vocab = {"<unk>": 0}
    idx = 1
    for char in sorted_chars:
        if char != "<unk>":
            vocab[char] = idx
            idx += 1

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=2, ensure_ascii=False)

    print(f"✅ Generated vocabulary saved to: {output_path}")
    return output_path


# ========== DATASET WITH BOTH CTC AND DECODER TARGETS ==========
class HFTransformerDataset(Dataset):
    """Dataset that returns both CTC and decoder targets"""

    def __init__(
        self,
        dataset,
        tokenizer,
        img_height=48,
        img_width=640,
        image_col="image",
        text_col="text",
    ):
        if load_dataset is None:
            raise ImportError("Please install 'datasets' library")

        self.dataset = dataset
        self.tokenizer = tokenizer
        self.img_height = img_height
        self.img_width = img_width
        self.image_col = image_col
        self.text_col = text_col

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        img = item[self.image_col]
        text = item[self.text_col]

        try:
            # Image preprocessing
            if img.mode != "L":
                img = img.convert("L")

            w, h = img.size
            new_w = int(w * self.img_height / h)
            img = img.resize((new_w, self.img_height), Image.BILINEAR)

            final_img = Image.new("L", (self.img_width, self.img_height), 128)
            paste_w = min(new_w, self.img_width)
            final_img.paste(img.crop((0, 0, paste_w, self.img_height)), (0, 0))

            img_tensor = torch.from_numpy(np.array(final_img)).float() / 255.0
            img_tensor = (img_tensor - 0.5) / 0.5
            img_tensor = img_tensor.unsqueeze(0)

            # ========== DECODER TARGETS ==========
            # [BOS, char1+offset, char2+offset, ..., EOS]
            dec_ids = []
            for c in text:
                raw_id = self.tokenizer.token_to_id.get(c, self.tokenizer.unk_id)
                dec_ids.append(raw_id + self.tokenizer.dec_offset)
            dec_ids = [self.tokenizer.dec_bos] + dec_ids + [self.tokenizer.dec_eos]

            # ========== CTC TARGETS ==========
            # [char1+ctc_offset, char2+ctc_offset, ...] (no BOS/EOS)
            ctc_ids = []
            for c in text:
                raw_id = self.tokenizer.token_to_id.get(c, self.tokenizer.unk_id)
                ctc_ids.append(raw_id + self.tokenizer.ctc_offset)

            return {
                "image": img_tensor,
                "dec_target": torch.LongTensor(dec_ids),
                "ctc_target": torch.LongTensor(ctc_ids),
                "ctc_target_len": len(ctc_ids),
                "text": text,
            }

        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            return {
                "image": torch.zeros(1, self.img_height, self.img_width),
                "dec_target": torch.LongTensor([1, 2]),  # BOS, EOS
                "ctc_target": torch.LongTensor([]),
                "ctc_target_len": 0,
                "text": "",
            }


class TransformerDataset(Dataset):
    """Local dataset that returns both CTC and decoder targets"""

    def __init__(self, labels_file, tokenizer, img_height=48, img_width=640):
        self.samples = []
        self.tokenizer = tokenizer
        self.img_height = img_height
        self.img_width = img_width

        labels_path = Path(labels_file)
        possible_img_dirs = [
            labels_path.parent / "images",
            labels_path.parent,
        ]

        with open(labels_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    img_name = parts[0]
                    text = parts[1]

                    for img_dir in possible_img_dirs:
                        img_path = img_dir / img_name
                        if img_path.exists():
                            self.samples.append((str(img_path), text))
                            break

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, text = self.samples[idx]

        try:
            img = Image.open(img_path).convert("L")
            w, h = img.size
            new_w = int(w * self.img_height / h)
            img = img.resize((new_w, self.img_height), Image.BILINEAR)

            final_img = Image.new("L", (self.img_width, self.img_height), 128)
            paste_w = min(new_w, self.img_width)
            final_img.paste(img.crop((0, 0, paste_w, self.img_height)), (0, 0))

            img_tensor = torch.from_numpy(np.array(final_img)).float() / 255.0
            img_tensor = (img_tensor - 0.5) / 0.5
            img_tensor = img_tensor.unsqueeze(0)

            # Decoder targets
            dec_ids = []
            for c in text:
                raw_id = self.tokenizer.token_to_id.get(c, self.tokenizer.unk_id)
                dec_ids.append(raw_id + self.tokenizer.dec_offset)
            dec_ids = [self.tokenizer.dec_bos] + dec_ids + [self.tokenizer.dec_eos]

            # CTC targets
            ctc_ids = []
            for c in text:
                raw_id = self.tokenizer.token_to_id.get(c, self.tokenizer.unk_id)
                ctc_ids.append(raw_id + self.tokenizer.ctc_offset)

            return {
                "image": img_tensor,
                "dec_target": torch.LongTensor(dec_ids),
                "ctc_target": torch.LongTensor(ctc_ids),
                "ctc_target_len": len(ctc_ids),
                "text": text,
            }

        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return {
                "image": torch.zeros(1, self.img_height, self.img_width),
                "dec_target": torch.LongTensor([1, 2]),
                "ctc_target": torch.LongTensor([]),
                "ctc_target_len": 0,
                "text": "",
            }


# Maximum sequence length for decoder to prevent OOM
MAX_DECODER_SEQ_LEN = 512  # Can be overridden via args


def collate_fn(batch, max_seq_len=MAX_DECODER_SEQ_LEN):
    """Collate function that handles both CTC and decoder targets"""
    images = torch.stack([item["image"] for item in batch])
    texts = [item["text"] for item in batch]

    # Decoder targets (padded and truncated to max_seq_len)
    dec_targets = [item["dec_target"] for item in batch]
    
    # Truncate sequences that exceed max_seq_len
    dec_targets_truncated = []
    for t in dec_targets:
        if len(t) > max_seq_len:
            # Keep BOS at start and truncate, ensuring we don't exceed max_seq_len
            dec_targets_truncated.append(t[:max_seq_len])
        else:
            dec_targets_truncated.append(t)
    
    max_dec_len = min(max(len(t) for t in dec_targets_truncated), max_seq_len)
    dec_padded = torch.zeros((len(batch), max_dec_len), dtype=torch.long)
    for i, t in enumerate(dec_targets_truncated):
        actual_len = min(len(t), max_dec_len)
        dec_padded[i, :actual_len] = t[:actual_len]

    # CTC targets (concatenated) - also truncate for consistency
    ctc_targets_list = []
    ctc_target_lens = []
    for item in batch:
        ctc_target = item["ctc_target"]
        # Truncate CTC target to max_seq_len - 2 (accounting for BOS/EOS in decoder)
        max_ctc_len = max_seq_len - 2
        if len(ctc_target) > max_ctc_len:
            ctc_target = ctc_target[:max_ctc_len]
        ctc_targets_list.append(ctc_target)
        ctc_target_lens.append(len(ctc_target))
    
    # Handle empty batch edge case
    if all(len(t) == 0 for t in ctc_targets_list):
        ctc_targets = torch.LongTensor([])
    else:
        ctc_targets = torch.cat([t for t in ctc_targets_list if len(t) > 0])
    
    ctc_target_lens = torch.LongTensor(ctc_target_lens)

    return {
        "images": images,
        "dec_targets": dec_padded,
        "ctc_targets": ctc_targets,
        "ctc_target_lens": ctc_target_lens,
        "texts": texts,
    }


def make_collate_fn(max_seq_len=MAX_DECODER_SEQ_LEN):
    """Create a collate function with custom max_seq_len"""
    def _collate(batch):
        return collate_fn(batch, max_seq_len=max_seq_len)
    return _collate


# ========== TRAINING LOOP WITH CTC + DECODER LOSS ==========
def train_command(args):
    print("=" * 60)
    print("  🚀 KiriOCR Transformer Training")
    print("=" * 60)

    device = args.device if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    # ========== 1. VOCAB ==========
    vocab_path = getattr(args, "vocab", None)
    hf_ds_train = None
    hf_ds_val = None

    if hasattr(args, "hf_dataset") and args.hf_dataset:
        print(f"\n📥 Loading HF dataset(s): {args.hf_dataset}")
        subset = getattr(args, "hf_subset", None)
        
        train_datasets = []
        val_datasets = []
        
        # Ensure it's a list (might be string if from config file)
        dataset_list = args.hf_dataset if isinstance(args.hf_dataset, list) else [args.hf_dataset]

        for ds_name in dataset_list:
            try:
                # Load training split
                ds_train = load_dataset(
                    ds_name, subset, split=args.hf_train_split
                )
                train_datasets.append(ds_train)
                print(f"   ✓ Loaded {ds_name} (train)")

                # Try to load validation split
                val_splits = [
                    getattr(args, "hf_val_split", None),
                    "validation",
                    "val",
                    "test",
                ]
                for split in val_splits:
                    if split:
                        try:
                            ds_val = load_dataset(ds_name, subset, split=split)
                            val_datasets.append(ds_val)
                            print(f"   ✓ Found validation split for {ds_name}: {split}")
                            break
                        except:
                            pass
            except Exception as e:
                print(f"❌ Error loading dataset {ds_name}: {e}")
        
        # Concatenate training datasets
        if train_datasets:
            if len(train_datasets) > 1 and concatenate_datasets:
                hf_ds_train = concatenate_datasets(train_datasets)
                print(f"   ✓ Concatenated {len(train_datasets)} training datasets")
            else:
                hf_ds_train = train_datasets[0]
        
        # Concatenate validation datasets
        if val_datasets:
            if len(val_datasets) > 1 and concatenate_datasets:
                hf_ds_val = concatenate_datasets(val_datasets)
                print(f"   ✓ Concatenated {len(val_datasets)} validation datasets")
            else:
                hf_ds_val = val_datasets[0]
        
        if not hf_ds_train:
             print("❌ No valid datasets loaded")
             return

    if not vocab_path or not os.path.exists(vocab_path):
        generated_vocab_path = os.path.join(args.output_dir, "vocab.json")

        if hf_ds_train:
            vocab_path = build_vocab_from_hf_dataset(
                hf_ds_train, generated_vocab_path, text_col=args.hf_text_col
            )
        elif hasattr(args, "train_labels") and args.train_labels:
            if os.path.exists(args.train_labels):
                vocab_path = build_vocab_from_dataset(
                    args.train_labels, generated_vocab_path
                )
            else:
                print(f"❌ Train labels not found: {args.train_labels}")
                return
        else:
            print("❌ No dataset provided")
            return

        if vocab_path is None:
            return

    # ========== 2. CONFIG & TOKENIZER ==========
    cfg = CFG()
    cfg.IMG_H = getattr(args, "height", 48)
    cfg.IMG_W = getattr(args, "width", 640)

    # ========== APPLY ARCHITECTURE CLI ARGS ==========
    # Encoder architecture
    if hasattr(args, "encoder_dim") and args.encoder_dim:
        cfg.ENC_DIM = args.encoder_dim
    if hasattr(args, "encoder_heads") and args.encoder_heads:
        cfg.ENC_HEADS = args.encoder_heads
    if hasattr(args, "encoder_layers") and args.encoder_layers:
        cfg.ENC_LAYERS = args.encoder_layers
    if hasattr(args, "encoder_ffn_dim") and args.encoder_ffn_dim:
        cfg.ENC_FF = args.encoder_ffn_dim

    # Decoder architecture
    if hasattr(args, "decoder_dim") and args.decoder_dim:
        cfg.DEC_DIM = args.decoder_dim
    if hasattr(args, "decoder_heads") and args.decoder_heads:
        cfg.DEC_HEADS = args.decoder_heads
    if hasattr(args, "decoder_layers") and args.decoder_layers:
        cfg.DEC_LAYERS = args.decoder_layers
    if hasattr(args, "decoder_ffn_dim") and args.decoder_ffn_dim:
        cfg.DEC_FF = args.decoder_ffn_dim

    # Regularization
    if hasattr(args, "dropout") and args.dropout is not None:
        cfg.DROPOUT = args.dropout

    # Validate architecture constraints
    if cfg.ENC_DIM % cfg.ENC_HEADS != 0:
        print(f"⚠️  Warning: encoder_dim ({cfg.ENC_DIM}) must be divisible by encoder_heads ({cfg.ENC_HEADS})")
        print(f"   Adjusting encoder_heads to {cfg.ENC_DIM // (cfg.ENC_DIM // cfg.ENC_HEADS)}")
        cfg.ENC_HEADS = max(1, cfg.ENC_DIM // (cfg.ENC_DIM // cfg.ENC_HEADS))
    
    if cfg.DEC_DIM % cfg.DEC_HEADS != 0:
        print(f"⚠️  Warning: decoder_dim ({cfg.DEC_DIM}) must be divisible by decoder_heads ({cfg.DEC_HEADS})")
        print(f"   Adjusting decoder_heads to {cfg.DEC_DIM // (cfg.DEC_DIM // cfg.DEC_HEADS)}")
        cfg.DEC_HEADS = max(1, cfg.DEC_DIM // (cfg.DEC_DIM // cfg.DEC_HEADS))

    try:
        tokenizer = CharTokenizer(vocab_path, cfg)
        print(f"\n📝 Vocabulary: {tokenizer.vocab_size} characters")
        print(f"   CTC classes: {tokenizer.ctc_classes}")
        print(f"   Decoder vocab: {tokenizer.dec_vocab}")
    except Exception as e:
        print(f"❌ Error loading tokenizer: {e}")
        return

    # ========== 3. MODEL ==========
    print(f"\n🏗️  Model Architecture:")
    print(f"   Encoder: dim={cfg.ENC_DIM}, heads={cfg.ENC_HEADS}, layers={cfg.ENC_LAYERS}, ffn={cfg.ENC_FF}")
    print(f"   Decoder: dim={cfg.DEC_DIM}, heads={cfg.DEC_HEADS}, layers={cfg.DEC_LAYERS}, ffn={cfg.DEC_FF}")
    print(f"   Dropout: {cfg.DROPOUT}")

    model = KiriOCR(cfg, tokenizer).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(
        f"   Total: {total_params:,} parameters ({total_params * 4 / 1024 / 1024:.1f} MB)"
    )

    # Load pretrained weights if specified
    if (
        hasattr(args, "from_model")
        and args.from_model
        and os.path.exists(args.from_model)
    ):
        print(f"   🔄 Loading weights from {args.from_model}")
        try:
            from_model_path = args.from_model
            # Check if it's a safetensors file
            if from_model_path.endswith('.safetensors') and HAS_SAFETENSORS:
                state_dict = load_file(from_model_path, device="cpu")
            else:
                # Load as torch checkpoint
                ckpt = torch.load(from_model_path, map_location="cpu", weights_only=False)
                state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
            model.load_state_dict(state_dict, strict=False)
            print("   ✓ Weights loaded")
        except Exception as e:
            print(f"   ⚠️ Could not load weights: {e}")

    # ========== 4. DATASETS ==========
    print(f"\n📂 Loading datasets...")

    if hf_ds_train:
        train_ds = HFTransformerDataset(
            hf_ds_train,
            tokenizer,
            img_height=cfg.IMG_H,
            img_width=cfg.IMG_W,
            image_col=args.hf_image_col,
            text_col=args.hf_text_col,
        )
        print(f"   Train: {len(train_ds)} samples")

        val_ds = None
        if hf_ds_val:
            val_ds = HFTransformerDataset(
                hf_ds_val,
                tokenizer,
                img_height=cfg.IMG_H,
                img_width=cfg.IMG_W,
                image_col=args.hf_image_col,
                text_col=args.hf_text_col,
            )
            print(f"   Val: {len(val_ds)} samples")
    else:
        train_ds = TransformerDataset(
            args.train_labels, tokenizer, img_height=cfg.IMG_H, img_width=cfg.IMG_W
        )
        print(f"   Train: {len(train_ds)} samples")

        val_ds = None
        if (
            hasattr(args, "val_labels")
            and args.val_labels
            and os.path.exists(args.val_labels)
        ):
            val_ds = TransformerDataset(
                args.val_labels, tokenizer, img_height=cfg.IMG_H, img_width=cfg.IMG_W
            )
            print(f"   Val: {len(val_ds)} samples")

    if len(train_ds) == 0:
        print("❌ No training samples found!")
        return

    # Get max sequence length from args (default to MAX_DECODER_SEQ_LEN)
    max_seq_len = getattr(args, "max_seq_len", MAX_DECODER_SEQ_LEN)
    print(f"   Max sequence length: {max_seq_len}")
    
    # Create collate function with the specified max_seq_len
    custom_collate = make_collate_fn(max_seq_len=max_seq_len)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=custom_collate,
        num_workers=1 if device == "cuda" else 0,
        pin_memory=(device == "cuda"),
    )

    val_loader = None
    if val_ds:
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=custom_collate,
            num_workers=4 if device == "cuda" else 0,
        )

    # ========== 5. LOSSES ==========
    # CTC Loss (for encoder)
    ctc_criterion = nn.CTCLoss(blank=tokenizer.blank_id, zero_infinity=True)

    # Cross-Entropy Loss (for decoder)
    ce_criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.dec_pad)

    # Loss weights
    ctc_weight = getattr(args, "ctc_weight", 0.5)
    dec_weight = getattr(args, "dec_weight", 0.5)
    print(f"\n⚖️  Loss weights: CTC={ctc_weight}, Decoder={dec_weight}")

    # ========== 6. OPTIMIZER & SCHEDULER ==========
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.98),
    )

    total_steps = len(train_loader) * args.epochs
    warmup_steps = min(4000, total_steps // 10)

    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=total_steps,
        pct_start=warmup_steps / total_steps,
        anneal_strategy="cos",
    )

    print(f"   Optimizer: AdamW (lr={args.lr})")
    print(f"   Scheduler: OneCycleLR (warmup={warmup_steps} steps)")

    # ========== 7. RESUME ==========
    start_epoch = 0
    global_step = 0
    best_val_acc = 0  # Track best validation accuracy (higher is better)

    # Try safetensors first, then fallback to .pt
    resume_paths = [
        f"{args.output_dir}/latest.safetensors",
        f"{args.output_dir}/latest.pt",
    ]
    
    if getattr(args, "resume", False):
        for resume_path in resume_paths:
            if os.path.exists(resume_path):
                print(f"\n🔄 Resuming from {resume_path}...")
                try:
                    ckpt = load_checkpoint(resume_path, device)
                    
                    # Load model weights
                    if ckpt.get("model") is not None:
                        model.load_state_dict(ckpt["model"], strict=False)
                        print(f"   ✓ Loaded model weights")
                    else:
                        print(f"   ⚠️ No model weights in checkpoint")

                    # Load optimizer state
                    if ckpt.get("optimizer") is not None:
                        try:
                            optimizer.load_state_dict(ckpt["optimizer"])
                            print(f"   ✓ Loaded optimizer state")
                        except Exception as e:
                            print(f"   ⚠️ Could not load optimizer: {e}")
                    else:
                        print(f"   ⚠️ No optimizer state in checkpoint")
                    
                    # Load scheduler state
                    if ckpt.get("scheduler") is not None:
                        try:
                            scheduler.load_state_dict(ckpt["scheduler"])
                            print(f"   ✓ Loaded scheduler state")
                        except Exception as e:
                            print(f"   ⚠️ Could not load scheduler: {e}")
                    else:
                        print(f"   ⚠️ No scheduler state in checkpoint")
                    
                    # Load training progress
                    if ckpt.get("epoch") is not None:
                        start_epoch = ckpt["epoch"]
                    if ckpt.get("step") is not None:
                        global_step = ckpt["step"]
                    # Handle both old "best_val_loss" and new "best_val_acc" keys
                    if ckpt.get("best_val_acc") is not None:
                        best_val_acc = ckpt["best_val_acc"]
                    elif ckpt.get("best_val_loss") is not None:
                        # Old checkpoints used best_val_loss (which was actually accuracy)
                        # If it's infinity, it means no best was ever saved
                        old_val = ckpt["best_val_loss"]
                        best_val_acc = 0 if old_val == float("inf") else old_val

                    print(f"   ✓ Resuming from epoch {start_epoch}, step {global_step}")
                    print(f"   ✓ Best val accuracy so far: {best_val_acc:.2f}%")
                    break
                except Exception as e:
                    print(f"   ⚠️ Resume failed: {e}")
                    import traceback
                    traceback.print_exc()

    # ========== 8. TRAINING LOOP ==========
    print(f"\n" + "=" * 60)
    print(f"  Starting Training: {args.epochs} epochs on {device}")
    print("=" * 60)

    history = {"train_loss": [], "val_loss": [], "ctc_loss": [], "dec_loss": []}

    # Info about resume state
    steps_per_epoch = len(train_loader)
    if global_step > 0:
        completed_in_epoch = global_step - (start_epoch * steps_per_epoch)
        print(f"\n📊 Resume info:")
        print(f"   Total steps completed: {global_step}")
        print(f"   Steps per epoch: {steps_per_epoch}")
        print(f"   Starting from epoch {start_epoch + 1} (will restart from beginning)")
        print(f"   Note: With shuffle=True, epoch restarts from batch 0 but LR continues correctly")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0
        epoch_ctc_loss = 0
        epoch_dec_loss = 0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for batch in pbar:
            imgs = batch["images"].to(device)
            dec_tgts = batch["dec_targets"].to(device)
            ctc_tgts = batch["ctc_targets"].to(device)
            ctc_lens = batch["ctc_target_lens"].to(device)

            optimizer.zero_grad()

            # ========== ENCODER ==========
            memory = model.encode(imgs)  # [B, T, D]

            # ========== CTC LOSS ==========
            ctc_logits = model.ctc_head(memory)  # [B, T, ctc_classes]
            ctc_logits = ctc_logits.permute(1, 0, 2)  # [T, B, C] for CTC
            ctc_log_probs = nn.functional.log_softmax(ctc_logits, dim=2)

            input_lens = torch.full(
                (imgs.size(0),), ctc_logits.size(0), dtype=torch.long, device=device
            )

            # Skip samples with empty targets
            valid_mask = ctc_lens > 0
            if valid_mask.sum() > 0:
                ctc_loss = ctc_criterion(
                    ctc_log_probs[:, valid_mask],
                    ctc_tgts,
                    input_lens[valid_mask],
                    ctc_lens[valid_mask],
                )
            else:
                ctc_loss = torch.tensor(0.0, device=device)

            # ========== DECODER LOSS ==========
            memory_proj = model.mem_proj(memory)

            dec_inp = dec_tgts[:, :-1]  # [BOS, a, b, c]
            dec_out = dec_tgts[:, 1:]  # [a, b, c, EOS]

            tgt_emb = model.dec_emb(dec_inp)
            # Apply positional encoding if available
            if model.dec_pos_enc is not None:
                tgt_emb = model.dec_pos_enc(tgt_emb)
            seq_len = dec_inp.size(1)
            tgt_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=device) * float("-inf"), diagonal=1
            )

            dec_output = model.dec(tgt=tgt_emb, memory=memory_proj, tgt_mask=tgt_mask)
            dec_logits = model.dec_head(model.dec_ln(dec_output))

            dec_loss = ce_criterion(
                dec_logits.reshape(-1, dec_logits.size(-1)), dec_out.reshape(-1)
            )

            # ========== COMBINED LOSS ==========
            loss = ctc_weight * ctc_loss + dec_weight * dec_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            # Track
            epoch_loss += loss.item()
            epoch_ctc_loss += ctc_loss.item()
            epoch_dec_loss += dec_loss.item()
            num_batches += 1
            global_step += 1

            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "ctc": f"{ctc_loss.item():.4f}",
                    "dec": f"{dec_loss.item():.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                }
            )

            # Save step checkpoint
            save_steps = getattr(args, "save_steps", 0)
            if save_steps > 0 and global_step % save_steps == 0:
                save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    cfg,
                    vocab_path,
                    epoch,
                    global_step,
                    best_val_acc,
                    f"{args.output_dir}/checkpoint_step_{global_step}.safetensors",
                )
                save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    cfg,
                    vocab_path,
                    epoch,
                    global_step,
                    best_val_acc,
                    f"{args.output_dir}/latest.safetensors",
                )

        # Epoch stats
        avg_loss = epoch_loss / max(1, num_batches)
        avg_ctc = epoch_ctc_loss / max(1, num_batches)
        avg_dec = epoch_dec_loss / max(1, num_batches)

        history["train_loss"].append(avg_loss)
        history["ctc_loss"].append(avg_ctc)
        history["dec_loss"].append(avg_dec)

        print(f"\n  Epoch {epoch+1} Summary:")
        print(
            f"    Train Loss: {avg_loss:.4f} (CTC: {avg_ctc:.4f}, Dec: {avg_dec:.4f})"
        )

        # ========== VALIDATION (OPTIMIZED) ==========
        val_loss = 0
        val_acc = 0
        val_dec_acc = 0

        if val_loader:
            model.eval()
            val_total = 0
            val_ctc_correct = 0
            val_dec_correct = 0
            
            # Limit validation samples for speed (sample ~10K from dataset)
            max_val_samples = getattr(args, "max_val_samples", None)
            
            from .model import compute_ctc_confidence

            with torch.no_grad():
                for batch_idx, batch in enumerate(tqdm(val_loader, desc="Validating", leave=False)):
                    imgs = batch["images"].to(device)
                    texts = batch["texts"]
                    batch_size = imgs.size(0)
                    
                    # Check if we've validated enough samples (only if limit is set)
                    if max_val_samples and val_total >= max_val_samples:
                        break

                    # ========== ENCODE ONCE for entire batch ==========
                    memory = model.encode(imgs)  # [B, T, D]
                    
                    # ========== BATCHED CTC VALIDATION ==========
                    ctc_logits = model.ctc_head(memory)  # [B, T, C]
                    
                    for i in range(batch_size):
                        if max_val_samples and val_total >= max_val_samples:
                            break
                            
                        try:
                            # CTC decode (fast - no autoregressive)
                            _, pred_ctc, _ = compute_ctc_confidence(
                                ctc_logits[i], tokenizer
                            )
                            if pred_ctc.strip() == texts[i].strip():
                                val_ctc_correct += 1
                        except Exception:
                            pass
                        val_total += 1
                    
                    # ========== DECODER VALIDATION (sampled) ==========
                    # Only validate decoder on first sample of each batch for speed
                    # Full decoder validation is too slow for large datasets
                    if batch_idx % 10 == 0:  # Sample every 10th batch
                        try:
                            from .model import beam_decode_one_batched
                            mem_proj = model.mem_proj(memory[:1])  # First sample only
                            ctc_logits_1 = ctc_logits[:1] if cfg.USE_CTC else None
                            
                            old_beam = cfg.BEAM
                            cfg.BEAM = 1  # Greedy for speed
                            pred_dec, _ = beam_decode_one_batched(
                                model, mem_proj, tokenizer, cfg, ctc_logits_1=ctc_logits_1
                            )
                            cfg.BEAM = old_beam
                            
                            if pred_dec.strip() == texts[0].strip():
                                val_dec_correct += 1
                        except Exception:
                            pass

            val_acc = val_ctc_correct / max(1, val_total) * 100
            # Decoder accuracy based on sampled batches
            sampled_batches = (batch_idx + 1) // 10 + 1
            val_dec_acc = val_dec_correct / max(1, sampled_batches) * 100
            
            print(f"    Val CTC Accuracy: {val_acc:.2f}% ({val_ctc_correct}/{val_total})")
            print(f"    Val Decoder Accuracy (sampled): {val_dec_acc:.2f}% ({val_dec_correct}/{sampled_batches} batches)")
            
            # Alert if decoder is significantly worse than CTC
            if val_acc - val_dec_acc > 15:
                print(f"    ⚠️  Decoder ({val_dec_acc:.1f}%) is underperforming CTC ({val_acc:.1f}%)")
                print(f"       Consider increasing --dec_weight (current effective loss balance)")

            history["val_loss"].append(val_acc)
            if "val_dec_acc" not in history:
                history["val_dec_acc"] = []
            history["val_dec_acc"].append(val_dec_acc)

        # Save epoch checkpoint
        save_checkpoint(
            model,
            optimizer,
            scheduler,
            cfg,
            vocab_path,
            epoch + 1,
            global_step,
            best_val_acc,
            f"{args.output_dir}/model_epoch_{epoch+1}.safetensors",
        )
        save_checkpoint(
            model,
            optimizer,
            scheduler,
            cfg,
            vocab_path,
            epoch + 1,
            global_step,
            best_val_acc,
            f"{args.output_dir}/latest.safetensors",
        )

        # Save best model (higher accuracy is better)
        if val_loader and val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                cfg,
                vocab_path,
                epoch + 1,
                global_step,
                best_val_acc,
                f"{args.output_dir}/model.safetensors",
            )
            print(f"    ✓ New best model! Acc: {val_acc:.2f}%")

        print()

    # Save training history
    with open(f"{args.output_dir}/history.json", "w") as f:
        json.dump(history, f, indent=2)

    print("=" * 60)
    print(f"  ✅ Training Complete!")
    print(f"     Models saved to: {args.output_dir}")
    print("=" * 60)


def save_checkpoint(
    model, optimizer, scheduler, cfg, vocab_path, epoch, step, best_val_acc, path
):
    """Save checkpoint in safetensors format with metadata JSON"""
    if HAS_SAFETENSORS and path.endswith('.safetensors'):
        # Save model weights as safetensors
        save_file(model.state_dict(), path)
        
        # Save metadata and optimizer state as JSON
        metadata_path = path.replace('.safetensors', '_meta.json')
        metadata = {
            "vocab_path": str(vocab_path),
            "epoch": epoch,
            "step": step,
            "best_val_acc": best_val_acc,
            "config": {
                # Image dimensions
                "IMG_H": cfg.IMG_H,
                "IMG_W": cfg.IMG_W,
                # Encoder architecture
                "ENC_DIM": cfg.ENC_DIM,
                "ENC_LAYERS": cfg.ENC_LAYERS,
                "ENC_HEADS": cfg.ENC_HEADS,
                "ENC_FF": cfg.ENC_FF,
                # Decoder architecture
                "DEC_DIM": cfg.DEC_DIM,
                "DEC_LAYERS": cfg.DEC_LAYERS,
                "DEC_HEADS": cfg.DEC_HEADS,
                "DEC_FF": cfg.DEC_FF,
                # Regularization
                "DROPOUT": cfg.DROPOUT,
                # Training flags
                "USE_CTC": cfg.USE_CTC,
                "USE_FP16": cfg.USE_FP16,
            }
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save optimizer/scheduler state separately (torch format for complex state)
        optim_path = path.replace('.safetensors', '_optim.pt')
        torch.save({
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        }, optim_path)
    else:
        # Fallback to torch.save
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "config": cfg,
                "vocab_path": vocab_path,
                "epoch": epoch,
                "step": step,
                "best_val_acc": best_val_acc,
            },
            path,
        )


def load_checkpoint(path, device="cpu"):
    """Load checkpoint from safetensors or pt format"""
    if path.endswith('.safetensors') and HAS_SAFETENSORS:
        # Load model weights
        print(f"   Loading safetensors checkpoint: {path}")
        model_state = load_file(path, device=str(device))
        
        # Load metadata
        metadata_path = path.replace('.safetensors', '_meta.json')
        metadata = {}
        if os.path.exists(metadata_path):
            print(f"   Loading metadata: {metadata_path}")
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        else:
            print(f"   ⚠️ Metadata file not found: {metadata_path}")
        
        # Load optimizer state
        optim_path = path.replace('.safetensors', '_optim.pt')
        optim_state = {}
        if os.path.exists(optim_path):
            print(f"   Loading optimizer state: {optim_path}")
            optim_state = torch.load(optim_path, map_location=device, weights_only=False)
        else:
            print(f"   ⚠️ Optimizer state file not found: {optim_path}")
        
        # Handle both old "best_val_loss" and new "best_val_acc" keys
        best_val = metadata.get("best_val_acc")
        if best_val is None:
            old_val = metadata.get("best_val_loss", 0)
            best_val = 0 if old_val == float("inf") else old_val
        
        result = {
            "model": model_state,
            "optimizer": optim_state.get("optimizer") if optim_state else None,
            "scheduler": optim_state.get("scheduler") if optim_state else None,
            "vocab_path": metadata.get("vocab_path", ""),
            "epoch": metadata.get("epoch", 0),
            "step": metadata.get("step", 0),
            "best_val_acc": best_val,
            "config": metadata.get("config", {}),
        }
        
        print(f"   Checkpoint info: epoch={result['epoch']}, step={result['step']}")
        return result
    else:
        # Load torch checkpoint
        print(f"   Loading torch checkpoint: {path}")
        ckpt = torch.load(path, map_location=device, weights_only=False)
        print(f"   Checkpoint keys: {list(ckpt.keys()) if isinstance(ckpt, dict) else 'state_dict only'}")
        return ckpt
