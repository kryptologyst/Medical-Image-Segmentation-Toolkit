"""Medical image segmentation evaluation metrics."""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.distance import directed_hausdorff


class SegmentationMetrics:
    """
    Comprehensive evaluation metrics for medical image segmentation.
    
    This class provides various metrics commonly used in medical image
    segmentation tasks including Dice, IoU, Hausdorff distance, and more.
    """
    
    def __init__(self, num_classes: int = 1, threshold: float = 0.5) -> None:
        """
        Initialize segmentation metrics.
        
        Args:
            num_classes: Number of classes (1 for binary segmentation).
            threshold: Threshold for binary classification.
        """
        self.num_classes = num_classes
        self.threshold = threshold
        self.reset()
    
    def reset(self) -> None:
        """Reset all accumulated metrics."""
        self.dice_scores: List[float] = []
        self.iou_scores: List[float] = []
        self.hausdorff_distances: List[float] = []
        self.asd_distances: List[float] = []
        self.sensitivity_scores: List[float] = []
        self.specificity_scores: List[float] = []
        self.precision_scores: List[float] = []
        self.recall_scores: List[float] = []
    
    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        """
        Update metrics with new predictions and targets.
        
        Args:
            pred: Predicted probabilities of shape (N, C, H, W).
            target: Ground truth masks of shape (N, C, H, W).
        """
        # Convert to numpy for metric computation
        pred_np = pred.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()
        
        # Apply threshold for binary segmentation
        pred_binary = (pred_np > self.threshold).astype(np.uint8)
        target_binary = target_np.astype(np.uint8)
        
        # Compute metrics for each sample in the batch
        for i in range(pred_binary.shape[0]):
            for c in range(self.num_classes):
                pred_c = pred_binary[i, c]
                target_c = target_binary[i, c]
                
                # Skip if no ground truth
                if target_c.sum() == 0:
                    continue
                
                # Compute metrics
                dice = self._compute_dice(pred_c, target_c)
                iou = self._compute_iou(pred_c, target_c)
                hausdorff = self._compute_hausdorff_distance(pred_c, target_c)
                asd = self._compute_asd(pred_c, target_c)
                sensitivity, specificity = self._compute_sensitivity_specificity(
                    pred_c, target_c
                )
                precision, recall = self._compute_precision_recall(pred_c, target_c)
                
                # Store metrics
                self.dice_scores.append(dice)
                self.iou_scores.append(iou)
                self.hausdorff_distances.append(hausdorff)
                self.asd_distances.append(asd)
                self.sensitivity_scores.append(sensitivity)
                self.specificity_scores.append(specificity)
                self.precision_scores.append(precision)
                self.recall_scores.append(recall)
    
    def _compute_dice(self, pred: np.ndarray, target: np.ndarray) -> float:
        """Compute Dice coefficient."""
        intersection = np.logical_and(pred, target).sum()
        union = pred.sum() + target.sum()
        
        if union == 0:
            return 1.0 if intersection == 0 else 0.0
        
        return 2.0 * intersection / union
    
    def _compute_iou(self, pred: np.ndarray, target: np.ndarray) -> float:
        """Compute Intersection over Union (IoU)."""
        intersection = np.logical_and(pred, target).sum()
        union = np.logical_or(pred, target).sum()
        
        if union == 0:
            return 1.0 if intersection == 0 else 0.0
        
        return intersection / union
    
    def _compute_hausdorff_distance(
        self, pred: np.ndarray, target: np.ndarray, percentile: float = 95.0
    ) -> float:
        """Compute Hausdorff distance."""
        try:
            # Get boundary points
            pred_points = self._get_boundary_points(pred)
            target_points = self._get_boundary_points(target)
            
            if len(pred_points) == 0 or len(target_points) == 0:
                return float('inf')
            
            # Compute directed Hausdorff distances
            h1 = directed_hausdorff(pred_points, target_points)[0]
            h2 = directed_hausdorff(target_points, pred_points)[0]
            
            return max(h1, h2)
        except Exception:
            return float('inf')
    
    def _compute_asd(self, pred: np.ndarray, target: np.ndarray) -> float:
        """Compute Average Surface Distance (ASD)."""
        try:
            # Get boundary points
            pred_points = self._get_boundary_points(pred)
            target_points = self._get_boundary_points(target)
            
            if len(pred_points) == 0 or len(target_points) == 0:
                return float('inf')
            
            # Compute distances from pred boundary to target boundary
            distances = []
            for p_point in pred_points:
                min_dist = min(
                    np.linalg.norm(p_point - t_point) for t_point in target_points
                )
                distances.append(min_dist)
            
            return np.mean(distances) if distances else float('inf')
        except Exception:
            return float('inf')
    
    def _get_boundary_points(self, mask: np.ndarray) -> List[np.ndarray]:
        """Get boundary points from a binary mask."""
        from scipy.ndimage import binary_erosion
        
        # Find boundary by eroding the mask
        eroded = binary_erosion(mask)
        boundary = mask & ~eroded
        
        # Get coordinates of boundary points
        points = np.where(boundary)
        return [np.array([y, x]) for y, x in zip(points[0], points[1])]
    
    def _compute_sensitivity_specificity(
        self, pred: np.ndarray, target: np.ndarray
    ) -> Tuple[float, float]:
        """Compute sensitivity and specificity."""
        tp = np.logical_and(pred, target).sum()
        tn = np.logical_and(~pred, ~target).sum()
        fp = np.logical_and(pred, ~target).sum()
        fn = np.logical_and(~pred, target).sum()
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        return sensitivity, specificity
    
    def _compute_precision_recall(
        self, pred: np.ndarray, target: np.ndarray
    ) -> Tuple[float, float]:
        """Compute precision and recall."""
        tp = np.logical_and(pred, target).sum()
        fp = np.logical_and(pred, ~target).sum()
        fn = np.logical_and(~pred, target).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        return precision, recall
    
    def compute(self) -> Dict[str, float]:
        """
        Compute final metrics.
        
        Returns:
            Dictionary containing all computed metrics.
        """
        if not self.dice_scores:
            return {}
        
        # Filter out infinite values for distance metrics
        valid_hausdorff = [h for h in self.hausdorff_distances if h != float('inf')]
        valid_asd = [a for a in self.asd_distances if a != float('inf')]
        
        metrics = {
            "dice_mean": np.mean(self.dice_scores),
            "dice_std": np.std(self.dice_scores),
            "iou_mean": np.mean(self.iou_scores),
            "iou_std": np.std(self.iou_scores),
            "sensitivity_mean": np.mean(self.sensitivity_scores),
            "sensitivity_std": np.std(self.sensitivity_scores),
            "specificity_mean": np.mean(self.specificity_scores),
            "specificity_std": np.std(self.specificity_scores),
            "precision_mean": np.mean(self.precision_scores),
            "precision_std": np.std(self.precision_scores),
            "recall_mean": np.mean(self.recall_scores),
            "recall_std": np.std(self.recall_scores),
        }
        
        # Add distance metrics if valid values exist
        if valid_hausdorff:
            metrics.update({
                "hausdorff_mean": np.mean(valid_hausdorff),
                "hausdorff_std": np.std(valid_hausdorff),
            })
        
        if valid_asd:
            metrics.update({
                "asd_mean": np.mean(valid_asd),
                "asd_std": np.std(valid_asd),
            })
        
        return metrics


def compute_dice_score(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """
    Compute Dice score for tensors.
    
    Args:
        pred: Predicted probabilities of shape (N, C, H, W).
        target: Ground truth masks of shape (N, C, H, W).
        smooth: Smoothing factor.
        
    Returns:
        Dice score tensor.
    """
    # Apply sigmoid if needed
    if pred.max() > 1.0 or pred.min() < 0.0:
        pred = torch.sigmoid(pred)
    
    # Flatten tensors
    pred_flat = pred.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)
    
    # Compute intersection and union
    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
    
    # Compute Dice coefficient
    dice = (2.0 * intersection + smooth) / (union + smooth)
    
    return dice


def compute_iou_score(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """
    Compute IoU score for tensors.
    
    Args:
        pred: Predicted probabilities of shape (N, C, H, W).
        target: Ground truth masks of shape (N, C, H, W).
        smooth: Smoothing factor.
        
    Returns:
        IoU score tensor.
    """
    # Apply sigmoid if needed
    if pred.max() > 1.0 or pred.min() < 0.0:
        pred = torch.sigmoid(pred)
    
    # Flatten tensors
    pred_flat = pred.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)
    
    # Compute intersection and union
    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1) - intersection
    
    # Compute IoU
    iou = (intersection + smooth) / (union + smooth)
    
    return iou
