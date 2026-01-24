"""
Training utilities for CRAFT text detection model.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import json
from PIL import Image
import cv2
from tqdm import tqdm

from .model import CRAFT


class CRAFTDataset(Dataset):
    """
    Dataset for CRAFT training.
    
    Loads images and their corresponding region/affinity maps.
    """
    
    def __init__(self, data_dir, image_size=512):
        self.data_dir = data_dir
        self.image_size = image_size
        
        info_path = os.path.join(data_dir, 'dataset_info.json')
        if not os.path.exists(info_path):
            raise FileNotFoundError(f"Dataset info not found at {info_path}")
            
        with open(info_path, 'r') as f:
            self.info = json.load(f)
            
        # Try loading samples from separate list file (new format)
        list_path = os.path.join(data_dir, 'annotations_list.json')
        if os.path.exists(list_path):
            with open(list_path, 'r') as f:
                self.samples = json.load(f)
        elif 'samples' in self.info:
            self.samples = self.info['samples']
        else:
            print(f"Error: 'samples' key missing in {info_path} and {list_path} not found.")
            print("It seems you are using an incompatible dataset format.")
            print("Please re-generate the dataset using the new generator:")
            print("  python -m kiri_ocr.cli generate-detector --text-file ... --num-train ...")
            raise ValueError("Incompatible dataset format. Please regenerate dataset.")
        print(f"Loaded {len(self.samples)} samples from {data_dir}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        img_path = os.path.join(self.data_dir, 'images', sample['image'])
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Create dummy
            image = Image.new('RGB', (self.image_size, 64))
        
        image = np.array(image).astype(np.float32) / 255.0
        
        # Normalize (ImageNet mean/std)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std
        
        image = image.transpose(2, 0, 1)  # CHW
        
        # Load maps
        region_path = os.path.join(self.data_dir, 'labels', sample['region_map'])
        affinity_path = os.path.join(self.data_dir, 'labels', sample['affinity_map'])
        
        try:
            region = np.load(region_path)
            affinity = np.load(affinity_path)
            
            # Resize maps to 1/2 size (CRAFT output size)
            h, w = region.shape
            region = cv2.resize(region, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST)
            affinity = cv2.resize(affinity, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST)
            
        except:
            h, w = image.shape[1] // 2, image.shape[2] // 2
            region = np.zeros((h, w), dtype=np.float32)
            affinity = np.zeros((h, w), dtype=np.float32)

        # Expand dims for channel
        region = np.expand_dims(region, 0)
        affinity = np.expand_dims(affinity, 0)
        
        return torch.FloatTensor(image), torch.FloatTensor(region), torch.FloatTensor(affinity)


class CRAFTTrainer:
    """
    Trainer for CRAFT text detection model.
    
    Example:
        trainer = CRAFTTrainer(data_dir='detector_dataset')
        trainer.train(epochs=100, batch_size=8)
    """
    
    def __init__(self, data_dir, output_dir='runs/detect/khmer_text_detector'):
        self.data_dir = data_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'weights'), exist_ok=True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.model = CRAFT(pretrained=True).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=1e-5)
        
    def train(self, epochs=100, batch_size=8, num_workers=2):
        """
        Train the CRAFT model.
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size for training
            num_workers: Number of data loading workers
        """
        dataset = CRAFTDataset(self.data_dir)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        
        print(f"Starting training for {epochs} epochs...")
        
        best_loss = float('inf')
        
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0
            
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            for images, regions, affinities in pbar:
                images = images.to(self.device)
                regions = regions.to(self.device)
                affinities = affinities.to(self.device)
                
                # Forward
                y, feature = self.model(images)
                # y is [B, H/2, W/2, 2] (channels last)
                
                # Permute to [B, 2, H/2, W/2]
                y = y.permute(0, 3, 1, 2)
                
                pred_region = y[:, 0:1, :, :]
                pred_affinity = y[:, 1:2, :, :]
                
                # Loss
                loss_region = self.criterion(torch.sigmoid(pred_region), regions)
                loss_affinity = self.criterion(torch.sigmoid(pred_affinity), affinities)
                loss = loss_region + loss_affinity
                
                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
            
            avg_loss = epoch_loss / len(dataloader)
            print(f"Epoch {epoch+1} Average Loss: {avg_loss:.6f}")
            
            # Save checkpoint
            if avg_loss < best_loss:
                best_loss = avg_loss
                save_path = os.path.join(self.output_dir, 'weights', 'best.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': best_loss,
                }, save_path)
                print(f"Saved best model to {save_path}")
                
            # Save last
            save_path = os.path.join(self.output_dir, 'weights', 'last.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': avg_loss,
            }, save_path)


def train_detector_command(args):
    """CLI Command handler for training."""
    # Args from cli.py: data_yaml (unused now), model_size, epochs, batch_size, etc.
    # We'll use --data-yaml to point to dataset directory (dataset_info.json is inside)
    # The default in CLI is detector_dataset/data.yaml. We can strip 'data.yaml'.
    
    data_path = args.data_yaml
    if data_path.endswith('data.yaml'):
        data_path = os.path.dirname(data_path)
    
    if not os.path.exists(data_path):
        # Fallback to detector_dataset
        if os.path.exists('detector_dataset'):
            data_path = 'detector_dataset'
        else:
            print(f"Error: Dataset not found at {data_path}")
            return

    trainer = CRAFTTrainer(
        data_dir=data_path,
        output_dir=f'runs/detect/{args.name}'
    )
    
    trainer.train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.workers if hasattr(args, 'workers') else 1
    )
