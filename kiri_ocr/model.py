import torch
import torch.nn as nn
from pathlib import Path

# ========== CHARACTER SET ==========
class CharacterSet:
    """Manages character to index mapping"""
    def __init__(self):
        self.chars = ['BLANK', 'PAD', 'SOS', ' ']  # Special tokens + Space
        self.char2idx = {'BLANK': 0, 'PAD': 1, 'SOS': 2, ' ': 3}
        self.idx2char = {0: 'BLANK', 1: 'PAD', 2: 'SOS', 3: ' '}
        
    def add_chars(self, text):
        """Add new characters from text"""
        for char in text:
            if char not in self.char2idx:
                idx = len(self.chars)
                self.chars.append(char)
                self.char2idx[char] = idx
                self.idx2char[idx] = char
    
    def encode(self, text):
        """Convert text to indices"""
        return [self.char2idx.get(char, 0) for char in text]
    
    def decode(self, indices):
        """Convert indices to text (CTC decode)"""
        chars = []
        prev_idx = None
        for idx in indices:
            if idx > 2 and idx != prev_idx:  # Skip blank, pad, sos
                chars.append(self.idx2char.get(idx, ''))
            prev_idx = idx
        return ''.join(chars)
    
    def __len__(self):
        return len(self.chars)
    
    def save(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            for char in self.chars:
                f.write(char + '\n')
    
    @classmethod
    def load(cls, path):
        charset = cls()
        with open(path, 'r', encoding='utf-8') as f:
            chars = [line.rstrip('\n') for line in f]
            charset.chars = chars
            charset.char2idx = {char: idx for idx, char in enumerate(chars)}
            charset.idx2char = {idx: char for idx, char in enumerate(chars)}
        return charset
    
    @classmethod
    def from_checkpoint(cls, checkpoint):
        """Load charset from model checkpoint dictionary"""
        charset = cls()
        if 'charset' in checkpoint:
            charset.chars = checkpoint['charset']
            charset.char2idx = {char: idx for idx, char in enumerate(charset.chars)}
            charset.idx2char = {idx: char for idx, char in enumerate(charset.chars)}
        return charset

def save_checkpoint(model, charset, optimizer, epoch, val_loss, accuracy, path):
    """Save model checkpoint with charset included"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'accuracy': accuracy,
        'charset': charset.chars
    }, path)

# ========== LIGHTWEIGHT OCR MODEL ==========
class LightweightOCR(nn.Module):
    """
    Lightweight OCR model (~13MB)
    Similar to the reference model you showed
    """
    def __init__(self, num_chars, hidden_size=256):
        super(LightweightOCR, self).__init__()
        
        # CNN backbone (lighter than before)
        self.cnn = nn.Sequential(
            # Block 1: [B, 1, 32, W] -> [B, 32, 32, W]
            self._conv_block(1, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),  # -> [B, 32, 16, W/2]
            
            # Block 2: [B, 32, 16, W/2] -> [B, 64, 16, W/2]
            self._conv_block(32, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),  # -> [B, 64, 8, W/4]
            
            # Block 3: [B, 64, 8, W/4] -> [B, 128, 8, W/4]
            self._conv_block(64, 128, kernel_size=3, stride=1, padding=1),
            self._conv_block(128, 128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # -> [B, 128, 4, W/4]
            
            # Block 4: [B, 128, 4, W/4] -> [B, 256, 4, W/4]
            self._conv_block(128, 256, kernel_size=3, stride=1, padding=1),
            self._conv_block(256, 256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=(4, 1), stride=(4, 1)),  # -> [B, 256, 1, W/4]
        )
        
        # Two-layer LSTM with intermediate projection
        self.lstm1 = nn.LSTM(
            256, hidden_size, 
            bidirectional=True, 
            batch_first=True
        )
        
        self.intermediate_linear = nn.Linear(hidden_size * 2, hidden_size)
        
        self.lstm2 = nn.LSTM(
            hidden_size, hidden_size, 
            bidirectional=True, 
            batch_first=True
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_size * 2, num_chars)
    
    def _conv_block(self, in_ch, out_ch, **kwargs):
        """Convolutional block with BatchNorm and ReLU"""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, bias=False, **kwargs),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        """
        Args:
            x: [B, 1, H, W]
        Returns:
            output: [T, B, num_chars]
        """
        # CNN features
        features = self.cnn(x)  # [B, 256, 1, W']
        
        # Remove height dimension
        features = features.squeeze(2)  # [B, 256, W']
        features = features.permute(0, 2, 1)  # [B, W', 256]
        
        # First LSTM layer
        rnn_out, _ = self.lstm1(features)  # [B, W', hidden*2]
        
        # Intermediate projection
        rnn_out = self.intermediate_linear(rnn_out)  # [B, W', hidden]
        rnn_out = torch.relu(rnn_out)
        
        # Second LSTM layer
        rnn_out, _ = self.lstm2(rnn_out)  # [B, W', hidden*2]
        
        # Fully connected
        logits = self.fc(rnn_out)  # [B, W', num_chars]
        
        # Permute for CTC
        return logits.permute(1, 0, 2)  # [W', B, num_chars]
