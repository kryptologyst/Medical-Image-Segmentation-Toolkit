"""Unit tests for medical image segmentation models."""

import pytest
import torch
import numpy as np

from src.models.unet import UNet, UNetPlusPlus
from src.losses.medical_losses import DiceLoss, TverskyLoss, FocalLoss, CombinedLoss
from src.metrics.segmentation_metrics import SegmentationMetrics, compute_dice_score, compute_iou_score
from src.utils.device import get_device, set_deterministic_seed
from src.data.dataset import SyntheticMedicalDataset


class TestUNet:
    """Test cases for U-Net model."""
    
    def test_unet_initialization(self):
        """Test U-Net model initialization."""
        model = UNet(in_channels=1, out_channels=1, base_features=64)
        assert model.in_channels == 1
        assert model.out_channels == 1
        assert model.base_features == 64
    
    def test_unet_forward_pass(self):
        """Test U-Net forward pass."""
        model = UNet(in_channels=1, out_channels=1)
        model.eval()
        
        # Create dummy input
        batch_size = 2
        height, width = 64, 64
        input_tensor = torch.randn(batch_size, 1, height, width)
        
        # Forward pass
        with torch.no_grad():
            output = model(input_tensor)
        
        # Check output shape
        assert output.shape == (batch_size, 1, height, width)
        assert output.dtype == torch.float32
    
    def test_unet_model_size(self):
        """Test U-Net model size computation."""
        model = UNet()
        size_info = model.get_model_size()
        
        assert 'total_parameters' in size_info
        assert 'trainable_parameters' in size_info
        assert 'model_size_mb' in size_info
        assert size_info['total_parameters'] > 0


class TestUNetPlusPlus:
    """Test cases for UNet++ model."""
    
    def test_unet_plus_plus_initialization(self):
        """Test UNet++ model initialization."""
        model = UNetPlusPlus(in_channels=1, out_channels=1, base_features=64)
        assert model.in_channels == 1
        assert model.out_channels == 1
        assert model.base_features == 64
    
    def test_unet_plus_plus_forward_pass(self):
        """Test UNet++ forward pass."""
        model = UNetPlusPlus(in_channels=1, out_channels=1)
        model.eval()
        
        # Create dummy input
        batch_size = 2
        height, width = 64, 64
        input_tensor = torch.randn(batch_size, 1, height, width)
        
        # Forward pass
        with torch.no_grad():
            output = model(input_tensor)
        
        # Check output shape
        assert output.shape == (batch_size, 1, height, width)
        assert output.dtype == torch.float32


class TestLossFunctions:
    """Test cases for loss functions."""
    
    def test_dice_loss(self):
        """Test Dice loss computation."""
        loss_fn = DiceLoss()
        
        # Create dummy predictions and targets
        pred = torch.randn(2, 1, 32, 32)
        target = torch.randint(0, 2, (2, 1, 32, 32)).float()
        
        loss = loss_fn(pred, target)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.shape == ()
        assert loss.item() >= 0.0
    
    def test_tversky_loss(self):
        """Test Tversky loss computation."""
        loss_fn = TverskyLoss(alpha=0.3, beta=0.7)
        
        # Create dummy predictions and targets
        pred = torch.randn(2, 1, 32, 32)
        target = torch.randint(0, 2, (2, 1, 32, 32)).float()
        
        loss = loss_fn(pred, target)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.shape == ()
        assert loss.item() >= 0.0
    
    def test_focal_loss(self):
        """Test Focal loss computation."""
        loss_fn = FocalLoss(alpha=1.0, gamma=2.0)
        
        # Create dummy predictions and targets
        pred = torch.randn(2, 1, 32, 32)
        target = torch.randint(0, 2, (2, 1, 32, 32)).float()
        
        loss = loss_fn(pred, target)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.shape == ()
        assert loss.item() >= 0.0
    
    def test_combined_loss(self):
        """Test Combined loss computation."""
        loss_fn = CombinedLoss(dice_weight=0.5, bce_weight=0.5)
        
        # Create dummy predictions and targets
        pred = torch.randn(2, 1, 32, 32)
        target = torch.randint(0, 2, (2, 1, 32, 32)).float()
        
        loss = loss_fn(pred, target)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.shape == ()
        assert loss.item() >= 0.0


class TestMetrics:
    """Test cases for evaluation metrics."""
    
    def test_segmentation_metrics_initialization(self):
        """Test SegmentationMetrics initialization."""
        metrics = SegmentationMetrics(num_classes=1, threshold=0.5)
        assert metrics.num_classes == 1
        assert metrics.threshold == 0.5
    
    def test_segmentation_metrics_update(self):
        """Test SegmentationMetrics update."""
        metrics = SegmentationMetrics()
        
        # Create dummy predictions and targets
        pred = torch.randn(2, 1, 32, 32)
        target = torch.randint(0, 2, (2, 1, 32, 32)).float()
        
        metrics.update(pred, target)
        
        # Check that metrics were updated
        assert len(metrics.dice_scores) > 0
        assert len(metrics.iou_scores) > 0
    
    def test_segmentation_metrics_compute(self):
        """Test SegmentationMetrics computation."""
        metrics = SegmentationMetrics()
        
        # Create dummy predictions and targets
        pred = torch.randn(2, 1, 32, 32)
        target = torch.randint(0, 2, (2, 1, 32, 32)).float()
        
        metrics.update(pred, target)
        results = metrics.compute()
        
        assert isinstance(results, dict)
        assert 'dice_mean' in results
        assert 'iou_mean' in results
    
    def test_dice_score_computation(self):
        """Test Dice score computation."""
        pred = torch.randn(2, 1, 32, 32)
        target = torch.randint(0, 2, (2, 1, 32, 32)).float()
        
        dice_scores = compute_dice_score(pred, target)
        
        assert dice_scores.shape == (2,)
        assert torch.all(dice_scores >= 0.0)
        assert torch.all(dice_scores <= 1.0)
    
    def test_iou_score_computation(self):
        """Test IoU score computation."""
        pred = torch.randn(2, 1, 32, 32)
        target = torch.randint(0, 2, (2, 1, 32, 32)).float()
        
        iou_scores = compute_iou_score(pred, target)
        
        assert iou_scores.shape == (2,)
        assert torch.all(iou_scores >= 0.0)
        assert torch.all(iou_scores <= 1.0)


class TestDeviceUtils:
    """Test cases for device utilities."""
    
    def test_get_device(self):
        """Test device detection."""
        device = get_device()
        assert isinstance(device, torch.device)
        assert device.type in ['cuda', 'mps', 'cpu']
    
    def test_set_deterministic_seed(self):
        """Test deterministic seeding."""
        set_deterministic_seed(42)
        
        # Test that seeding works
        torch.manual_seed(42)
        rand1 = torch.randn(10)
        
        torch.manual_seed(42)
        rand2 = torch.randn(10)
        
        assert torch.allclose(rand1, rand2)


class TestDataset:
    """Test cases for dataset classes."""
    
    def test_synthetic_dataset_initialization(self):
        """Test SyntheticMedicalDataset initialization."""
        dataset = SyntheticMedicalDataset(size=(64, 64), num_samples=100)
        assert len(dataset) == 100
        assert dataset.size == (64, 64)
    
    def test_synthetic_dataset_getitem(self):
        """Test SyntheticMedicalDataset __getitem__."""
        dataset = SyntheticMedicalDataset(size=(32, 32), num_samples=10)
        
        image, mask = dataset[0]
        
        assert isinstance(image, torch.Tensor)
        assert isinstance(mask, torch.Tensor)
        assert image.shape == (1, 32, 32)
        assert mask.shape == (1, 32, 32)
        assert image.dtype == torch.float32
        assert mask.dtype == torch.float32
    
    def test_synthetic_dataset_consistency(self):
        """Test SyntheticMedicalDataset consistency."""
        dataset = SyntheticMedicalDataset(size=(32, 32), num_samples=5)
        
        # Get the same item multiple times
        image1, mask1 = dataset[0]
        image2, mask2 = dataset[0]
        
        # Should be identical (deterministic generation)
        assert torch.allclose(image1, image2)
        assert torch.allclose(mask1, mask2)


class TestIntegration:
    """Integration tests."""
    
    def test_training_step(self):
        """Test a single training step."""
        # Create model
        model = UNet(in_channels=1, out_channels=1)
        
        # Create loss function
        loss_fn = DiceLoss()
        
        # Create optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Create dummy data
        images = torch.randn(2, 1, 32, 32)
        masks = torch.randint(0, 2, (2, 1, 32, 32)).float()
        
        # Training step
        model.train()
        optimizer.zero_grad()
        
        outputs = model(images)
        loss = loss_fn(outputs, masks)
        
        loss.backward()
        optimizer.step()
        
        # Check that loss is computed
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0.0
    
    def test_evaluation_step(self):
        """Test a single evaluation step."""
        # Create model
        model = UNet(in_channels=1, out_channels=1)
        model.eval()
        
        # Create metrics
        metrics = SegmentationMetrics()
        
        # Create dummy data
        images = torch.randn(2, 1, 32, 32)
        masks = torch.randint(0, 2, (2, 1, 32, 32)).float()
        
        # Evaluation step
        with torch.no_grad():
            outputs = model(images)
            predictions = torch.sigmoid(outputs)
            
            metrics.update(predictions, masks)
        
        # Check that metrics were updated
        assert len(metrics.dice_scores) > 0


if __name__ == "__main__":
    pytest.main([__file__])
