import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import numpy as np
import json

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

# Import the model definition
from .model_transformer import HybridContextOCRV2, CFG, CharTokenizer

# --- 1. AUTO VOCAB BUILDER ---
def build_vocab_from_hf_dataset(dataset, output_path, text_col="text"):
    """Scans a HF dataset to create vocab_char.json automatically."""
    print(f"📖 Scanning HF Dataset to build vocabulary...")
    unique_chars = set()
    
    # optimize: select only text column
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
    
    # Sort for deterministic order
    sorted_chars = sorted(list(unique_chars))
    
    # Create mapping: <unk> is 0, others follow
    # Note: Model reserves 0-3 for special tokens internally, 
    # but the JSON just needs to map chars to IDs. 
    # The CharTokenizer class handles the offset logic.
    vocab = {"<unk>": 0}
    
    idx = 1
    for char in sorted_chars:
        # Avoid overwriting <unk> if it exists in text
        if char != "<unk>":
            vocab[char] = idx
            idx += 1
            
    # Save to JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Generated vocabulary saved to: {output_path}")
    return output_path

# --- 2. DATASET HANDLING ---
class HFTransformerDataset(Dataset):
    def __init__(self, dataset, tokenizer, img_height=48, img_width=640, image_col="image", text_col="text"):
        if load_dataset is None:
            raise ImportError("Please install 'datasets' library: pip install datasets")
            
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
            if img.mode != "L":
                img = img.convert("L")
            
            w, h = img.size
            
            # Resize height to 48, keep aspect ratio
            new_w = int(w * self.img_height / h)
            img = img.resize((new_w, self.img_height), Image.BILINEAR)
            
            # Create canvas (padding)
            final_img = Image.new("L", (self.img_width, self.img_height), 128) # 128 is gray padding
            paste_w = min(new_w, self.img_width)
            final_img.paste(img.crop((0,0, paste_w, self.img_height)), (0, 0))
            
            # Normalize [-1, 1]
            img_tensor = torch.from_numpy(np.array(final_img)).float() / 255.0
            img_tensor = (img_tensor - 0.5) / 0.5
            img_tensor = img_tensor.unsqueeze(0) # [1, H, W]

            # Tokenize
            ids = [self.tokenizer.token_to_id.get(c, self.tokenizer.unk_id) for c in text]
            # Add BOS (1) and EOS (2) tokens
            ids = [self.tokenizer.dec_bos] + ids + [self.tokenizer.dec_eos]
            
            return img_tensor, torch.LongTensor(ids)
            
        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            return torch.zeros(1, self.img_height, self.img_width), torch.LongTensor([1, 2])

class TransformerDataset(Dataset):
    def __init__(self, labels_file, tokenizer, img_height=48, img_width=640):
        self.samples = []
        self.tokenizer = tokenizer
        self.img_height = img_height
        self.img_width = img_width
        
        # Smart path resolution for images
        labels_path = Path(labels_file)
        possible_img_dirs = [
            labels_path.parent / "images", # data/train/images
            labels_path.parent,            # data/train/
        ]

        with open(labels_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    img_name = parts[0]
                    text = parts[1]
                    
                    found = False
                    for img_dir in possible_img_dirs:
                        img_path = img_dir / img_name
                        if img_path.exists():
                            self.samples.append((str(img_path), text))
                            found = True
                            break
                    
                    # Optional: Print warning if image missing (commented out to reduce noise)
                    # if not found: print(f"Warning: Missing {img_name}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, text = self.samples[idx]
        
        try:
            img = Image.open(img_path).convert("L")
            w, h = img.size
            
            # Resize height to 48, keep aspect ratio
            new_w = int(w * self.img_height / h)
            img = img.resize((new_w, self.img_height), Image.BILINEAR)
            
            # Create canvas (padding)
            final_img = Image.new("L", (self.img_width, self.img_height), 128) # 128 is gray padding
            paste_w = min(new_w, self.img_width)
            final_img.paste(img.crop((0,0, paste_w, self.img_height)), (0, 0))
            
            # Normalize [-1, 1]
            img_tensor = torch.from_numpy(np.array(final_img)).float() / 255.0
            img_tensor = (img_tensor - 0.5) / 0.5
            img_tensor = img_tensor.unsqueeze(0) # [1, H, W]

            # Tokenize
            ids = [self.tokenizer.token_to_id.get(c, self.tokenizer.unk_id) for c in text]
            # Add BOS (1) and EOS (2) tokens
            ids = [self.tokenizer.dec_bos] + ids + [self.tokenizer.dec_eos]
            
            return img_tensor, torch.LongTensor(ids)
            
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return torch.zeros(1, self.img_height, self.img_width), torch.LongTensor([1, 2])

def collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack(images)
    
    # Dynamic padding for text targets
    max_len = max([len(t) for t in targets])
    padded_targets = torch.zeros((len(targets), max_len), dtype=torch.long)
    for i, t in enumerate(targets):
        padded_targets[i, :len(t)] = t
        
    return images, padded_targets

# --- 3. TRAINING LOOP ---
def train_command(args):
    print(f"🚀 Initializing Transformer Training...")
    
    device = args.device if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. HANDLE VOCAB (The new auto logic)
    vocab_path = args.vocab
    
    # If vocab not provided OR file doesn't exist, create it
    if not vocab_path or not os.path.exists(vocab_path):
        generated_vocab_path = os.path.join(args.output_dir, "vocab_auto.json")
        vocab_path = build_vocab_from_dataset(args.train_labels, generated_vocab_path)
        if vocab_path is None:
            print("❌ Failed to generate vocabulary. Exiting.")
            return

    # 2. Config & Tokenizer
    cfg = CFG()
    cfg.IMG_H = args.height 
    
    try:
        tokenizer = CharTokenizer(vocab_path, cfg)
        print(f"  Vocabulary Size: {tokenizer.vocab_size} characters")
    except Exception as e:
        print(f"❌ Error loading tokenizer: {e}")
        return

    # 3. Model
    model = HybridContextOCRV2(cfg, tokenizer).to(device)
    
    # Load Pretrained (Optional)
    if args.from_model and os.path.exists(args.from_model):
        print(f"  🔄 Loading weights from {args.from_model}")
        try:
            ckpt = torch.load(args.from_model, map_location='cpu')
            state_dict = ckpt['model'] if 'model' in ckpt else ckpt
            model.load_state_dict(state_dict, strict=False)
            print("  ✓ Weights loaded (strict=False)")
        except Exception as e:
            print(f"  ⚠️ Could not load weights: {e}")
            print("  ⚠️ Training from random initialization.")

    # 4. Datasets
    train_loader = None
    
    if hasattr(args, 'hf_dataset') and args.hf_dataset:
        print(f"  ⬇️ Loading HF dataset: {args.hf_dataset}")
        subset = getattr(args, "hf_subset", None)
        try:
            ds = load_dataset(args.hf_dataset, subset, split=args.hf_train_split)
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return

        # Auto-build vocab if needed
        if not os.path.exists(vocab_path) and vocab_path == args.vocab:
             # Only if user provided specific vocab path that doesn't exist, we fail?
             # Or if we defaulted to auto, we build.
             # Earlier code block handled file-based vocab build.
             # We need to handle HF-based vocab build if vocab_path refers to the auto path.
             if "vocab_auto.json" in vocab_path:
                 vocab_path = build_vocab_from_hf_dataset(ds, vocab_path, text_col=args.hf_text_col)
                 # Re-init tokenizer with new vocab
                 tokenizer = CharTokenizer(vocab_path, cfg)
        
        train_ds = HFTransformerDataset(
            ds,
            tokenizer,
            img_height=args.height,
            image_col=args.hf_image_col,
            text_col=args.hf_text_col
        )
        print(f"  Loaded {len(train_ds)} training samples from HF.")
        
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=4 if device == "cuda" else 0
        )
        
    elif args.train_labels:
        train_ds = TransformerDataset(args.train_labels, tokenizer, img_height=args.height)
        if len(train_ds) == 0:
            print("❌ No images found. Check your labels file paths.")
            return
            
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=4 if device == "cuda" else 0
        )
        print(f"  Loaded {len(train_ds)} training samples.")
    else:
        print("❌ Error: --train-labels or --hf-dataset is required.")
        return

    # 5. Optimizer
    # Ignore padding (0) in loss
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.dec_pad)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 6. Training Loop
    print(f"  Starting training for {args.epochs} epochs on {device}...")
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        count = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for imgs, tgts in pbar:
            imgs, tgts = imgs.to(device), tgts.to(device)
            
            # Prepare Inputs/Targets for Teacher Forcing
            # Input:  <bos> A B C
            # Target: A B C <eos>
            dec_inp = tgts[:, :-1] 
            dec_out = tgts[:, 1:]  
            
            optimizer.zero_grad()
            
            # --- Forward Pass ---
            memory = model.encode(imgs)
            memory_proj = model.mem_proj(memory)
            
            tgt_emb = model.dec_emb(dec_inp)
            
            # Causal Mask (prevent looking ahead)
            seq_len = dec_inp.size(1)
            tgt_mask = torch.triu(torch.ones(seq_len, seq_len, device=device) * float('-inf'), diagonal=1)
            
            out = model.dec(tgt=tgt_emb, memory=memory_proj, tgt_mask=tgt_mask)
            
            # Project to Vocab
            logits = model.dec_head(model.dec_ln(out))
            
            # Calculate Loss
            # Flatten [Batch, Seq, Vocab] -> [Batch*Seq, Vocab]
            loss = criterion(logits.reshape(-1, logits.size(-1)), dec_out.reshape(-1))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            count += 1
            pbar.set_postfix({'loss': loss.item()})
            
        # Save Checkpoint
        avg_loss = epoch_loss / max(1, count)
        print(f"  Epoch {epoch+1} Avg Loss: {avg_loss:.4f}")
        
        # Save model and the vocab used for it
        save_path = f"{args.output_dir}/model_epoch_{epoch+1}.pt"
        torch.save({
            'model': model.state_dict(), 
            'config': cfg,
            'vocab_path': vocab_path # Store where the vocab is
        }, save_path)
        
        # Save 'latest.pt' for easy resumption
        torch.save(model.state_dict(), f"{args.output_dir}/latest.pt")

    print(f"✅ Training Complete. Models saved to {args.output_dir}")