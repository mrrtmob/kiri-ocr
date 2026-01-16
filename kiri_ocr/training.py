import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

from .model import LightweightOCR, CharacterSet, save_checkpoint


# ========== DATASET ==========
class OCRDataset(Dataset):
    def __init__(self, labels_file, charset, img_height=32):
        self.img_height = img_height
        self.charset = charset
        self.samples = []

        labels_path = Path(labels_file)
        self.img_dir = labels_path.parent / "images"

        with open(labels_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 2:
                    img_name, text = parts
                    charset.add_chars(text)
                    self.samples.append((img_name, text))

        print(f"  Loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, text = self.samples[idx]
        img_path = self.img_dir / img_name

        img = Image.open(img_path).convert("L")

        # Resize maintaining aspect ratio
        w, h = img.size
        new_h = self.img_height
        new_w = int(w * new_h / h)
        img = img.resize((new_w, new_h), Image.LANCZOS)

        # Normalize
        img = np.array(img).astype(np.float32) / 255.0
        img = torch.FloatTensor(img).unsqueeze(0)

        target = self.charset.encode(text)
        target = torch.LongTensor(target)

        return img, target, text


class HFOCRDataset(Dataset):
    def __init__(
        self, dataset, charset, img_height=32, image_col="image", text_col="text"
    ):
        if load_dataset is None:
            raise ImportError("Please install 'datasets' library: pip install datasets")

        self.img_height = img_height
        self.charset = charset
        self.image_col = image_col
        self.text_col = text_col
        self.dataset = dataset

        # Iterate to build charset
        print(f"  🔄 Scanning charset from '{text_col}' column...")
        if text_col not in self.dataset.column_names:
            raise ValueError(
                f"Column '{text_col}' not found in dataset. Available: {self.dataset.column_names}"
            )

        # Optimize: select only text column to avoid decoding images during scan
        try:
            iter_ds = self.dataset.select_columns([text_col])
        except:
            iter_ds = self.dataset

        for item in tqdm(iter_ds, desc="Scanning charset"):
            text = item.get(text_col, "")
            if text:
                charset.add_chars(text)
        
        # Clear memory
        del iter_ds
        import gc
        gc.collect()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        img = item[self.image_col]
        text = item[self.text_col]

        if img.mode != "L":
            img = img.convert("L")

        # Resize maintaining aspect ratio
        w, h = img.size
        new_h = self.img_height
        new_w = int(w * new_h / h)
        img = img.resize((new_w, new_h), Image.LANCZOS)

        # Normalize
        img = np.array(img).astype(np.float32) / 255.0
        img = torch.FloatTensor(img).unsqueeze(0)

        target = self.charset.encode(text)
        target = torch.LongTensor(target)

        return img, target, text


def collate_fn(batch):
    images, targets, texts = zip(*batch)

    max_width = max([img.size(2) for img in images])
    batch_size = len(images)
    height = images[0].size(1)

    padded_images = torch.zeros(batch_size, 1, height, max_width)

    for i, img in enumerate(images):
        w = img.size(2)
        padded_images[i, :, :, :w] = img

    target_lengths = torch.LongTensor([len(t) for t in targets])
    targets_concat = torch.cat(targets)

    return padded_images, targets_concat, target_lengths, texts


def train_model(
    model,
    train_loader,
    val_loader,
    charset,
    num_epochs=200,
    device="cuda",
    save_dir="models",
    lr=0.001,
    weight_decay=0.0001,
):
    os.makedirs(save_dir, exist_ok=True)

    model = model.to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )

    best_loss = float("inf")
    best_acc = 0

    print("\n" + "=" * 70)
    print("  🚀 Training Lightweight OCR")
    print("=" * 70)

    history = {"train_loss": [], "val_loss": [], "acc": []}

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0

        # Progress bar for training
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")

        for batch_idx, (images, targets, target_lengths, texts) in enumerate(pbar):
            images = images.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            input_lengths = torch.full(
                size=(outputs.size(1),),
                fill_value=outputs.size(0),
                dtype=torch.long,
                device=device,
            )

            log_probs = nn.functional.log_softmax(outputs, dim=2)
            loss = criterion(log_probs, targets, input_lengths, target_lengths)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            train_loss += loss.item()

            # Update progress bar
            pbar.set_postfix({"loss": loss.item()})

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, targets, target_lengths, texts in val_loader:
                images = images.to(device)
                targets = targets.to(device)
                target_lengths = target_lengths.to(device)

                outputs = model(images)

                input_lengths = torch.full(
                    size=(outputs.size(1),),
                    fill_value=outputs.size(0),
                    dtype=torch.long,
                    device=device,
                )

                log_probs = nn.functional.log_softmax(outputs, dim=2)
                loss = criterion(log_probs, targets, input_lengths, target_lengths)
                val_loss += loss.item()

                # Accuracy
                _, preds = outputs.max(2)
                preds = preds.transpose(0, 1)

                for i, text in enumerate(texts):
                    pred_indices = preds[i].cpu().numpy()
                    pred_text = charset.decode(pred_indices)
                    if pred_text == text:
                        correct += 1
                    total += 1

        val_loss /= len(val_loader)
        accuracy = correct / total * 100

        scheduler.step(val_loss)

        print(
            f"Epoch [{epoch+1:3d}/{num_epochs}] "
            f"Train: {train_loss:.4f} | "
            f"Val: {val_loss:.4f} | "
            f"Acc: {accuracy:5.2f}%"
        )

        # Store history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["acc"].append(accuracy)

        # Save checkpoint every epoch
        save_checkpoint(
            model,
            charset,
            optimizer,
            epoch + 1,
            val_loss,
            accuracy,
            f"{save_dir}/checkpoint_epoch_{epoch+1}.kiri",
        )

        # Save best model
        if accuracy > best_acc or (accuracy == best_acc and val_loss < best_loss):
            best_loss = val_loss
            best_acc = accuracy

            save_checkpoint(
                model,
                charset,
                optimizer,
                epoch + 1,
                val_loss,
                accuracy,
                f"{save_dir}/model.kiri",
            )
            print(f"  ✓ Saved Best! Acc: {accuracy:.2f}%")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    # Plot history
    try:
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(history["train_loss"], label="Train Loss")
        plt.plot(history["val_loss"], label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(history["acc"], label="Validation Accuracy", color="green")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.title("Training Accuracy")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(f"{save_dir}/training_history.png")
        print(f"\n📊 Training plot saved to {save_dir}/training_history.png")
    except Exception as e:
        print(f"\n⚠️ Failed to save plot: {e}")

    print(f"\n🎉 Training complete! Best accuracy: {best_acc:.2f}%\n")


def train_command(args):
    IMAGE_HEIGHT = args.height
    BATCH_SIZE = args.batch_size
    NUM_EPOCHS = args.epochs
    HIDDEN_SIZE = args.hidden_size

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("⚠️ CUDA not available, using CPU")
        device = "cpu"

    print(f"\n🖥️  Device: {device}\n")

    # Load charset from pretrained model if available
    charset = CharacterSet()
    if (
        hasattr(args, "from_model")
        and args.from_model
        and os.path.exists(args.from_model)
    ):
        try:
            checkpoint = torch.load(args.from_model, map_location="cpu")
            if "charset" in checkpoint:
                print(f"🔄 Loading charset from {args.from_model}")
                charset = CharacterSet.from_checkpoint(checkpoint)
        except Exception:
            pass  # Will be handled later or just start fresh

    print("📂 Loading datasets...")

    train_dataset = None
    val_dataset = None

    if hasattr(args, "hf_dataset") and args.hf_dataset:
        print(f"  ⬇️ Loading HF dataset: {args.hf_dataset}")
        subset = getattr(args, "hf_subset", None)

        # Load train split
        try:
            ds = load_dataset(args.hf_dataset, subset, split=args.hf_train_split)
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return

        val_ds = None
        # Try finding val split
        val_splits = (
            [args.hf_val_split] if args.hf_val_split else ["val", "test", "validation"]
        )

        for split in val_splits:
            if not split:
                continue
            try:
                # We check if we can load it
                candidate = load_dataset(args.hf_dataset, subset, split=split)
                val_ds = candidate
                print(f"  ✓ Found validation split: '{split}'")
                break
            except:
                pass

        if val_ds is None:
            print(
                f"  ⚠️ No validation split found. Splitting {args.hf_val_percent*100}% from train..."
            )
            try:
                split_ds = ds.train_test_split(test_size=args.hf_val_percent)
                ds = split_ds["train"]
                val_ds = split_ds["test"]
            except Exception as e:
                print(f"  ⚠️ Could not split dataset: {e}. Using train for validation.")
                val_ds = ds

        train_dataset = HFOCRDataset(
            ds,
            charset,
            img_height=IMAGE_HEIGHT,
            image_col=args.hf_image_col,
            text_col=args.hf_text_col,
        )
        val_dataset = HFOCRDataset(
            val_ds,
            charset,
            img_height=IMAGE_HEIGHT,
            image_col=args.hf_image_col,
            text_col=args.hf_text_col,
        )
    else:
        if not os.path.exists(args.train_labels):
            print(f"❌ Training labels not found: {args.train_labels}")
            return

        train_dataset = OCRDataset(args.train_labels, charset, img_height=IMAGE_HEIGHT)

        if args.val_labels and os.path.exists(args.val_labels):
            val_dataset = OCRDataset(args.val_labels, charset, img_height=IMAGE_HEIGHT)
        else:
            print(
                f"⚠️ Validation labels not found. Using training set for validation (not recommended)."
            )
            val_dataset = train_dataset

    print(f"\n📊 Dataset: {len(train_dataset)} train, {len(val_dataset)} val")
    print(f"📝 Characters: {len(charset)}\n")
    
    # Clear memory before training
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    os.makedirs(args.output_dir, exist_ok=True)
    # charset.save(f'{args.output_dir}/charset_lite.txt') # Not needed if using .kiri

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4 if device == "cuda" else 0,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4 if device == "cuda" else 0,
        collate_fn=collate_fn,
    )

    model = LightweightOCR(num_chars=len(charset), hidden_size=HIDDEN_SIZE)

    # Load pretrained model if specified
    if hasattr(args, "from_model") and args.from_model:
        print(f"🔄 Loading pretrained model from {args.from_model}")
        if os.path.exists(args.from_model):
            try:
                checkpoint = torch.load(args.from_model, map_location=device)
                if "model_state_dict" in checkpoint:
                    state_dict = checkpoint["model_state_dict"]
                    model_dict = model.state_dict()

                    # Filter out unnecessary keys or shape mismatches
                    pretrained_dict = {
                        k: v
                        for k, v in state_dict.items()
                        if k in model_dict and v.shape == model_dict[k].shape
                    }

                    # Log what was skipped
                    skipped_keys = [k for k in state_dict if k not in pretrained_dict]
                    if skipped_keys:
                        print(
                            f"  ⚠️ Skipped mismatched layers (fine-tuning): {skipped_keys[:5]} ..."
                        )

                    # Overwrite entries in the existing state dict
                    model_dict.update(pretrained_dict)
                    model.load_state_dict(model_dict)
                    print("  ✓ Weights loaded successfully")
                else:
                    print("  ⚠️ Invalid checkpoint format (missing model_state_dict)")
            except Exception as e:
                print(f"  ❌ Error loading model: {e}")
        else:
            print(f"  ❌ Pretrained model not found: {args.from_model}")

    total_params = sum(p.numel() for p in model.parameters())
    model_size_mb = total_params * 4 / 1024 / 1024

    print(f"🏗️  Model: {total_params:,} params ({model_size_mb:.2f} MB)\n")

    train_model(
        model,
        train_loader,
        val_loader,
        charset,
        num_epochs=NUM_EPOCHS,
        device=device,
        save_dir=args.output_dir,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
