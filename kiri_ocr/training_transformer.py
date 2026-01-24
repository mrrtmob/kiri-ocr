# kiri_ocr/training_transformer.py
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

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

from .model_transformer import KiriOCR, CFG, CharTokenizer, greedy_ctc_decode


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


def collate_fn(batch):
    """Collate function that handles both CTC and decoder targets"""
    images = torch.stack([item["image"] for item in batch])
    texts = [item["text"] for item in batch]

    # Decoder targets (padded)
    dec_targets = [item["dec_target"] for item in batch]
    max_dec_len = max(len(t) for t in dec_targets)
    dec_padded = torch.zeros((len(batch), max_dec_len), dtype=torch.long)
    for i, t in enumerate(dec_targets):
        dec_padded[i, : len(t)] = t

    # CTC targets (concatenated)
    ctc_targets = torch.cat([item["ctc_target"] for item in batch])
    ctc_target_lens = torch.LongTensor([item["ctc_target_len"] for item in batch])

    return {
        "images": images,
        "dec_targets": dec_padded,
        "ctc_targets": ctc_targets,
        "ctc_target_lens": ctc_target_lens,
        "texts": texts,
    }


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
        print(f"\n📥 Loading HF dataset: {args.hf_dataset}")
        subset = getattr(args, "hf_subset", None)
        try:
            hf_ds_train = load_dataset(
                args.hf_dataset, subset, split=args.hf_train_split
            )

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
                        hf_ds_val = load_dataset(args.hf_dataset, subset, split=split)
                        print(f"   ✓ Found validation split: {split}")
                        break
                    except:
                        pass
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return

    if not vocab_path or not os.path.exists(vocab_path):
        generated_vocab_path = os.path.join(args.output_dir, "vocab_auto.json")

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

    try:
        tokenizer = CharTokenizer(vocab_path, cfg)
        print(f"\n📝 Vocabulary: {tokenizer.vocab_size} characters")
        print(f"   CTC classes: {tokenizer.ctc_classes}")
        print(f"   Decoder vocab: {tokenizer.dec_vocab}")
    except Exception as e:
        print(f"❌ Error loading tokenizer: {e}")
        return

    # ========== 3. MODEL ==========
    model = KiriOCR(cfg, tokenizer).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(
        f"\n🏗️  Model: {total_params:,} parameters ({total_params * 4 / 1024 / 1024:.1f} MB)"
    )

    # Load pretrained weights if specified
    if (
        hasattr(args, "from_model")
        and args.from_model
        and os.path.exists(args.from_model)
    ):
        print(f"   🔄 Loading weights from {args.from_model}")
        try:
            ckpt = torch.load(args.from_model, map_location="cpu", weights_only=False)
            state_dict = ckpt["model"] if "model" in ckpt else ckpt
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

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4 if device == "cuda" else 0,
        pin_memory=(device == "cuda"),
    )

    val_loader = None
    if val_ds:
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
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
    best_val_loss = float("inf")

    resume_path = f"{args.output_dir}/latest.pt"
    if getattr(args, "resume", False) and os.path.exists(resume_path):
        print(f"\n🔄 Resuming from {resume_path}...")
        try:
            ckpt = torch.load(resume_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model"], strict=False)

            if "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
            if "scheduler" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler"])
            if "epoch" in ckpt:
                start_epoch = ckpt["epoch"]
            if "step" in ckpt:
                global_step = ckpt["step"]
            if "best_val_loss" in ckpt:
                best_val_loss = ckpt["best_val_loss"]

            print(f"   ✓ Resumed at epoch {start_epoch}, step {global_step}")
        except Exception as e:
            print(f"   ⚠️ Resume failed: {e}")

    # ========== 8. TRAINING LOOP ==========
    print(f"\n" + "=" * 60)
    print(f"  Starting Training: {args.epochs} epochs on {device}")
    print("=" * 60)

    history = {"train_loss": [], "val_loss": [], "ctc_loss": [], "dec_loss": []}

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
                    best_val_loss,
                    f"{args.output_dir}/checkpoint_step_{global_step}.pt",
                )
                save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    cfg,
                    vocab_path,
                    epoch,
                    global_step,
                    best_val_loss,
                    f"{args.output_dir}/latest.pt",
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

        # ========== VALIDATION ==========
        val_loss = 0
        val_acc = 0

        if val_loader:
            model.eval()
            val_total = 0
            val_correct = 0

            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validating", leave=False):
                    imgs = batch["images"].to(device)
                    texts = batch["texts"]

                    # Use CTC greedy decode for validation
                    for i in range(imgs.size(0)):
                        try:
                            pred_text, conf = greedy_ctc_decode(
                                model, imgs[i : i + 1], tokenizer, cfg
                            )
                            if pred_text.strip() == texts[i].strip():
                                val_correct += 1
                        except:
                            pass
                        val_total += 1

            val_acc = val_correct / max(1, val_total) * 100
            print(f"    Val Accuracy: {val_acc:.2f}% ({val_correct}/{val_total})")

            history["val_loss"].append(val_acc)

        # Save epoch checkpoint
        save_checkpoint(
            model,
            optimizer,
            scheduler,
            cfg,
            vocab_path,
            epoch + 1,
            global_step,
            best_val_loss,
            f"{args.output_dir}/model_epoch_{epoch+1}.pt",
        )
        save_checkpoint(
            model,
            optimizer,
            scheduler,
            cfg,
            vocab_path,
            epoch + 1,
            global_step,
            best_val_loss,
            f"{args.output_dir}/latest.pt",
        )

        # Save best model
        if val_loader and val_acc > best_val_loss:
            best_val_loss = val_acc
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                cfg,
                vocab_path,
                epoch + 1,
                global_step,
                best_val_loss,
                f"{args.output_dir}/best_model.pt",
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
    model, optimizer, scheduler, cfg, vocab_path, epoch, step, best_val_loss, path
):
    """Save full checkpoint"""
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "config": cfg,
            "vocab_path": vocab_path,
            "epoch": epoch,
            "step": step,
            "best_val_loss": best_val_loss,
        },
        path,
    )
