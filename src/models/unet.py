"""Modern U-Net architecture for medical image segmentation."""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Convolutional block with BatchNorm and ReLU activation."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        """
        Initialize convolutional block.
        
        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Size of the convolutional kernel.
        """
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=kernel_size // 2
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the convolutional block."""
        return self.relu(self.bn(self.conv(x)))


class DoubleConv(nn.Module):
    """Double convolutional block used in U-Net."""
    
    def __init__(self, in_channels: int, out_channels: int) -> None:
        """
        Initialize double convolutional block.
        
        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
        """
        super().__init__()
        self.conv1 = ConvBlock(in_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the double convolutional block."""
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class UNet(nn.Module):
    """
    U-Net architecture for medical image segmentation.
    
    This implementation follows the original U-Net paper with modern improvements
    including batch normalization and proper skip connections.
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_features: int = 64,
        dropout_rate: float = 0.1,
    ) -> None:
        """
        Initialize U-Net model.
        
        Args:
            in_channels: Number of input channels (e.g., 1 for grayscale, 3 for RGB).
            out_channels: Number of output channels (number of classes).
            base_features: Number of base features in the first layer.
            dropout_rate: Dropout rate for regularization.
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Encoder (downsampling path)
        self.enc1 = DoubleConv(in_channels, base_features)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(base_features, base_features * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(base_features * 2, base_features * 4)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = DoubleConv(base_features * 4, base_features * 8)
        self.pool4 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = DoubleConv(base_features * 8, base_features * 16)
        self.dropout = nn.Dropout2d(dropout_rate)
        
        # Decoder (upsampling path)
        self.upconv4 = nn.ConvTranspose2d(base_features * 16, base_features * 8, 2, 2)
        self.dec4 = DoubleConv(base_features * 16, base_features * 8)
        
        self.upconv3 = nn.ConvTranspose2d(base_features * 8, base_features * 4, 2, 2)
        self.dec3 = DoubleConv(base_features * 8, base_features * 4)
        
        self.upconv2 = nn.ConvTranspose2d(base_features * 4, base_features * 2, 2, 2)
        self.dec2 = DoubleConv(base_features * 4, base_features * 2)
        
        self.upconv1 = nn.ConvTranspose2d(base_features * 2, base_features, 2, 2)
        self.dec1 = DoubleConv(base_features * 2, base_features)
        
        # Final classification layer
        self.final = nn.Conv2d(base_features, out_channels, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the U-Net.
        
        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width).
            
        Returns:
            Output tensor of shape (batch_size, out_channels, height, width).
        """
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        
        # Bottleneck
        b = self.bottleneck(self.pool4(e4))
        b = self.dropout(b)
        
        # Decoder with skip connections
        d4 = self.upconv4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.upconv3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        # Final output
        output = self.final(d1)
        
        return output
    
    def get_model_size(self) -> dict:
        """
        Get model size information.
        
        Returns:
            Dictionary containing model size metrics.
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": total_params * 4 / (1024 * 1024),  # Assuming float32
        }


class UNetPlusPlus(nn.Module):
    """
    UNet++ architecture with nested skip connections.
    
    This implementation provides better gradient flow and feature reuse
    compared to the standard U-Net.
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_features: int = 64,
        dropout_rate: float = 0.1,
    ) -> None:
        """
        Initialize UNet++ model.
        
        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            base_features: Number of base features in the first layer.
            dropout_rate: Dropout rate for regularization.
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Encoder layers
        self.enc1 = DoubleConv(in_channels, base_features)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(base_features, base_features * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(base_features * 2, base_features * 4)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = DoubleConv(base_features * 4, base_features * 8)
        self.pool4 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = DoubleConv(base_features * 8, base_features * 16)
        self.dropout = nn.Dropout2d(dropout_rate)
        
        # UNet++ nested skip connections
        self.upconv4 = nn.ConvTranspose2d(base_features * 16, base_features * 8, 2, 2)
        self.dec4_0 = DoubleConv(base_features * 16, base_features * 8)
        self.dec4_1 = DoubleConv(base_features * 12, base_features * 8)
        self.dec4_2 = DoubleConv(base_features * 10, base_features * 8)
        self.dec4_3 = DoubleConv(base_features * 9, base_features * 8)
        
        self.upconv3 = nn.ConvTranspose2d(base_features * 8, base_features * 4, 2, 2)
        self.dec3_0 = DoubleConv(base_features * 8, base_features * 4)
        self.dec3_1 = DoubleConv(base_features * 6, base_features * 4)
        self.dec3_2 = DoubleConv(base_features * 5, base_features * 4)
        
        self.upconv2 = nn.ConvTranspose2d(base_features * 4, base_features * 2, 2, 2)
        self.dec2_0 = DoubleConv(base_features * 4, base_features * 2)
        self.dec2_1 = DoubleConv(base_features * 3, base_features * 2)
        
        self.upconv1 = nn.ConvTranspose2d(base_features * 2, base_features, 2, 2)
        self.dec1_0 = DoubleConv(base_features * 2, base_features)
        
        # Final classification layer
        self.final = nn.Conv2d(base_features, out_channels, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the UNet++.
        
        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width).
            
        Returns:
            Output tensor of shape (batch_size, out_channels, height, width).
        """
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        
        # Bottleneck
        b = self.bottleneck(self.pool4(e4))
        b = self.dropout(b)
        
        # UNet++ decoder with nested skip connections
        d4_0 = self.upconv4(b)
        d4_0 = torch.cat([d4_0, e4], dim=1)
        d4_0 = self.dec4_0(d4_0)
        
        d4_1 = self.upconv3(d4_0)
        d4_1 = torch.cat([d4_1, e3, d4_0], dim=1)
        d4_1 = self.dec4_1(d4_1)
        
        d4_2 = self.upconv2(d4_1)
        d4_2 = torch.cat([d4_2, e2, d4_1], dim=1)
        d4_2 = self.dec4_2(d4_2)
        
        d4_3 = self.upconv1(d4_2)
        d4_3 = torch.cat([d4_3, e1, d4_2], dim=1)
        d4_3 = self.dec4_3(d4_3)
        
        # Final output
        output = self.final(d4_3)
        
        return output
