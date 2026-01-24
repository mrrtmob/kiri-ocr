"""
CRAFT (Character Region Awareness for Text detection) Model.

This module contains the neural network architecture for CRAFT-based text detection.
"""
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def init_weights(modules):
    """Initialize network weights."""
    for m in modules:
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()


class vgg16_bn(nn.Module):
    """VGG16 with Batch Normalization backbone for CRAFT."""
    
    def __init__(self, pretrained=True, freeze=True):
        super(vgg16_bn, self).__init__()
        # Load pretrained vgg16_bn from torchvision
        from torchvision.models import vgg16_bn as tv_vgg16_bn, VGG16_BN_Weights
        if pretrained:
            vgg_pretrained = tv_vgg16_bn(weights=VGG16_BN_Weights.DEFAULT)
        else:
            vgg_pretrained = tv_vgg16_bn(weights=None)
            
        self.features = vgg_pretrained.features
        
        # We need to access intermediate layers, so we slice them
        # VGG16_BN features structure:
        # 0-5: Block 1 (64) - conv,bn,relu, conv,bn,relu
        # 6: MaxPool
        # 7-12: Block 2 (128)
        # 13: MaxPool
        # 14-22: Block 3 (256) - conv,bn,relu, conv,bn,relu, conv,bn,relu
        # 23: MaxPool
        # 24-32: Block 4 (512)
        # 33: MaxPool
        # 34-42: Block 5 (512)
        # 43: MaxPool
        
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        
        for x in range(13):         self.slice1.add_module(str(x), self.features[x]) # to relu2_2 (index 12)
        for x in range(13, 23):     self.slice2.add_module(str(x), self.features[x]) # to relu3_3 (index 22)
        for x in range(23, 33):     self.slice3.add_module(str(x), self.features[x]) # to relu4_3 (index 32)
        for x in range(33, 43):     self.slice4.add_module(str(x), self.features[x]) # to relu5_3 (index 42)
        
        # FC6 and FC7 equivalent layers (dilated)
        self.slice5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        
        init_weights(self.slice5.modules())

        if freeze:
            for param in self.slice1.parameters(): param.requires_grad = False
            for param in self.slice2.parameters(): param.requires_grad = False
            # usually only freeze early layers

    def forward(self, x):
        h = self.slice1(x)
        h_relu2_2 = h
        h = self.slice2(h)
        h_relu3_3 = h
        h = self.slice3(h)
        h_relu4_3 = h
        h = self.slice4(h)
        h_relu5_3 = h
        h = self.slice5(h)
        h_fc7 = h
        
        # Returns [conv7, conv5_3, conv4_3, conv3_3, conv2_2]
        return [h_fc7, h_relu5_3, h_relu4_3, h_relu3_3, h_relu2_2]


class double_conv(nn.Module):
    """Double convolution block for U-Net style decoder."""
    
    def __init__(self, in_ch, mid_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + mid_ch, mid_ch, kernel_size=1),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class CRAFT(nn.Module):
    """
    CRAFT: Character Region Awareness for Text detection.
    
    This model outputs two maps:
    - Region score map: probability of each pixel being part of a character
    - Affinity score map: probability of each pixel being between adjacent characters
    """
    
    def __init__(self, pretrained=False, freeze=False):
        super(CRAFT, self).__init__()

        # Base network (VGG16 with batch norm)
        self.basenet = vgg16_bn(pretrained, freeze)

        # U network decoder
        self.upconv1 = double_conv(1024, 512, 256)
        self.upconv2 = double_conv(512, 256, 128)
        self.upconv3 = double_conv(256, 128, 64)
        self.upconv4 = double_conv(128, 64, 32)

        num_class = 2
        self.conv_cls = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, num_class, kernel_size=1),
        )

        init_weights(self.upconv1.modules())
        init_weights(self.upconv2.modules())
        init_weights(self.upconv3.modules())
        init_weights(self.upconv4.modules())
        init_weights(self.conv_cls.modules())
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input image tensor [B, 3, H, W]
            
        Returns:
            y: Output scores [B, H/2, W/2, 2] (region, affinity)
            feature: Feature map from last upconv layer
        """
        # Base network
        sources = self.basenet(x)

        # U network decoder
        y = torch.cat([sources[0], sources[1]], dim=1)
        y = self.upconv1(y)

        y = F.interpolate(y, size=sources[2].size()[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources[2]], dim=1)
        y = self.upconv2(y)

        y = F.interpolate(y, size=sources[3].size()[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources[3]], dim=1)
        y = self.upconv3(y)

        y = F.interpolate(y, size=sources[4].size()[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources[4]], dim=1)
        feature = self.upconv4(y)

        y = self.conv_cls(feature)

        return y.permute(0, 2, 3, 1), feature


class CRAFTDetector:
    """
    High-level wrapper for CRAFT text detection.
    
    Example:
        detector = CRAFTDetector()
        detector.load_weights('best.pth')
        boxes = detector.detect_text('image.jpg')
    """
    
    def __init__(self, pretrained=False):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = CRAFT(pretrained=pretrained).to(self.device)
        self.model.eval()
    
    def load_weights(self, path):
        """Load model weights from checkpoint."""
        if not os.path.exists(path):
            print(f"Warning: Model path {path} does not exist")
            return
            
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            print(f"Weights loaded from {path}")
            self.model.eval()
        except Exception as e:
            print(f"Error loading weights: {e}")

    def detect_text(self, image_path, text_threshold=0.7, link_threshold=0.4, low_text=0.4, poly=False):
        """
        Detect text in an image.
        
        Args:
            image_path: Path to input image
            text_threshold: Text confidence threshold
            link_threshold: Link confidence threshold
            low_text: Low text score threshold
            poly: Whether to return polygon outputs
            
        Returns:
            List of bounding boxes
        """
        from . import imgproc
        from . import utils as craft_utils
        
        # Load image
        image = imgproc.loadImage(image_path)
        
        # Run inference
        bboxes, polys, score_text = self.test_net(image, text_threshold, link_threshold, low_text, poly)
        
        return bboxes

    def test_net(self, image, text_threshold, link_threshold, low_text, poly):
        """Run inference on an image."""
        from . import imgproc
        from . import utils as craft_utils
        
        # resize
        canvas_size = 1280
        mag_ratio = 1.5
        img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(
            image, canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=mag_ratio
        )
        ratio_h = ratio_w = 1 / target_ratio

        # preprocessing
        x = imgproc.normalizeMeanVariance(img_resized)
        x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
        x = x.unsqueeze(0)                          # [c, h, w] to [b, c, h, w]
        x = x.to(self.device)

        # forward pass
        with torch.no_grad():
            y, feature = self.model(x)

        # make score and link map
        score_text = y[0, :, :, 0].cpu().data.numpy()
        score_link = y[0, :, :, 1].cpu().data.numpy()
        
        # Apply sigmoid (since we trained with logits)
        y = torch.sigmoid(y)
        score_text = y[0, :, :, 0].cpu().data.numpy()
        score_link = y[0, :, :, 1].cpu().data.numpy()

        # Post-processing
        boxes, polys = craft_utils.getDetBoxes(
            score_text, score_link, text_threshold, link_threshold, low_text, poly
        )

        # coordinate adjustment
        boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
        polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
        for k in range(len(polys)):
            if polys[k] is None:
                polys[k] = boxes[k]

        return boxes, polys, score_text
