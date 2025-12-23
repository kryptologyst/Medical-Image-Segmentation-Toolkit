"""Medical image segmentation loss functions."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice loss for medical image segmentation.
    
    The Dice coefficient measures the overlap between predicted and ground truth
    segmentation masks. This loss function is particularly effective for
    imbalanced segmentation tasks.
    """
    
    def __init__(self, smooth: float = 1e-6, reduction: str = "mean") -> None:
        """
        Initialize Dice loss.
        
        Args:
            smooth: Smoothing factor to avoid division by zero.
            reduction: Reduction method ('mean', 'sum', 'none').
        """
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute Dice loss.
        
        Args:
            pred: Predicted probabilities of shape (N, C, H, W).
            target: Ground truth masks of shape (N, C, H, W).
            
        Returns:
            Dice loss value.
        """
        # Apply sigmoid to predictions if not already applied
        if pred.max() > 1.0 or pred.min() < 0.0:
            pred = torch.sigmoid(pred)
        
        # Flatten tensors
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        
        # Compute intersection and union
        intersection = (pred_flat * target_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
        
        # Compute Dice coefficient
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        # Convert to loss (1 - dice)
        loss = 1.0 - dice
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class TverskyLoss(nn.Module):
    """
    Tversky loss for medical image segmentation.
    
    The Tversky index is a generalization of the Dice coefficient that allows
    for different weights on false positives and false negatives.
    """
    
    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        smooth: float = 1e-6,
        reduction: str = "mean",
    ) -> None:
        """
        Initialize Tversky loss.
        
        Args:
            alpha: Weight for false positives.
            beta: Weight for false negatives.
            smooth: Smoothing factor to avoid division by zero.
            reduction: Reduction method ('mean', 'sum', 'none').
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute Tversky loss.
        
        Args:
            pred: Predicted probabilities of shape (N, C, H, W).
            target: Ground truth masks of shape (N, C, H, W).
            
        Returns:
            Tversky loss value.
        """
        # Apply sigmoid to predictions if not already applied
        if pred.max() > 1.0 or pred.min() < 0.0:
            pred = torch.sigmoid(pred)
        
        # Flatten tensors
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        
        # Compute true positives, false positives, and false negatives
        tp = (pred_flat * target_flat).sum(dim=1)
        fp = (pred_flat * (1 - target_flat)).sum(dim=1)
        fn = ((1 - pred_flat) * target_flat).sum(dim=1)
        
        # Compute Tversky index
        tversky = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )
        
        # Convert to loss (1 - tversky)
        loss = 1.0 - tversky
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class FocalLoss(nn.Module):
    """
    Focal loss for addressing class imbalance in medical image segmentation.
    
    Focal loss down-weights easy examples and focuses on hard examples,
    making it particularly effective for imbalanced datasets.
    """
    
    def __init__(
        self,
        alpha: float = 1.0,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        """
        Initialize Focal loss.
        
        Args:
            alpha: Weighting factor for rare class.
            gamma: Focusing parameter.
            reduction: Reduction method ('mean', 'sum', 'none').
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute Focal loss.
        
        Args:
            pred: Predicted logits of shape (N, C, H, W).
            target: Ground truth masks of shape (N, C, H, W).
            
        Returns:
            Focal loss value.
        """
        # Compute binary cross entropy
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        
        # Compute focal weight
        pt = torch.exp(-bce)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        
        # Apply focal weight
        focal_loss = focal_weight * bce
        
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class CombinedLoss(nn.Module):
    """
    Combined loss function for medical image segmentation.
    
    Combines multiple loss functions to leverage their complementary strengths.
    """
    
    def __init__(
        self,
        dice_weight: float = 0.5,
        bce_weight: float = 0.5,
        focal_weight: float = 0.0,
        tversky_weight: float = 0.0,
        dice_smooth: float = 1e-6,
        focal_alpha: float = 1.0,
        focal_gamma: float = 2.0,
        tversky_alpha: float = 0.3,
        tversky_beta: float = 0.7,
    ) -> None:
        """
        Initialize combined loss.
        
        Args:
            dice_weight: Weight for Dice loss.
            bce_weight: Weight for Binary Cross Entropy loss.
            focal_weight: Weight for Focal loss.
            tversky_weight: Weight for Tversky loss.
            dice_smooth: Smoothing factor for Dice loss.
            focal_alpha: Alpha parameter for Focal loss.
            focal_gamma: Gamma parameter for Focal loss.
            tversky_alpha: Alpha parameter for Tversky loss.
            tversky_beta: Beta parameter for Tversky loss.
        """
        super().__init__()
        
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.focal_weight = focal_weight
        self.tversky_weight = tversky_weight
        
        # Initialize loss functions
        self.dice_loss = DiceLoss(smooth=dice_smooth)
        self.bce_loss = nn.BCEWithLogitsLoss()
        
        if focal_weight > 0:
            self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        else:
            self.focal_loss = None
            
        if tversky_weight > 0:
            self.tversky_loss = TverskyLoss(
                alpha=tversky_alpha, beta=tversky_beta, smooth=dice_smooth
            )
        else:
            self.tversky_loss = None
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute combined loss.
        
        Args:
            pred: Predicted logits of shape (N, C, H, W).
            target: Ground truth masks of shape (N, C, H, W).
            
        Returns:
            Combined loss value.
        """
        total_loss = 0.0
        
        # Dice loss
        if self.dice_weight > 0:
            dice_loss = self.dice_loss(pred, target)
            total_loss += self.dice_weight * dice_loss
        
        # Binary Cross Entropy loss
        if self.bce_weight > 0:
            bce_loss = self.bce_loss(pred, target)
            total_loss += self.bce_weight * bce_loss
        
        # Focal loss
        if self.focal_weight > 0 and self.focal_loss is not None:
            focal_loss = self.focal_loss(pred, target)
            total_loss += self.focal_weight * focal_loss
        
        # Tversky loss
        if self.tversky_weight > 0 and self.tversky_loss is not None:
            tversky_loss = self.tversky_loss(pred, target)
            total_loss += self.tversky_weight * tversky_loss
        
        return total_loss


def get_loss_function(loss_name: str, **kwargs) -> nn.Module:
    """
    Get loss function by name.
    
    Args:
        loss_name: Name of the loss function.
        **kwargs: Additional arguments for the loss function.
        
    Returns:
        Loss function instance.
        
    Raises:
        ValueError: If loss_name is not recognized.
    """
    loss_functions = {
        "dice": DiceLoss,
        "tversky": TverskyLoss,
        "focal": FocalLoss,
        "bce": nn.BCEWithLogitsLoss,
        "combined": CombinedLoss,
    }
    
    if loss_name not in loss_functions:
        raise ValueError(f"Unknown loss function: {loss_name}")
    
    return loss_functions[loss_name](**kwargs)
