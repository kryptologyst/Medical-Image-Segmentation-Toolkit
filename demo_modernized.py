#!/usr/bin/env python3
"""
Modern Medical Image Segmentation Demo

This script demonstrates the modernized medical image segmentation toolkit
with U-Net architecture, comprehensive evaluation, and explainability features.

This is a research demonstration tool and is NOT intended for clinical use.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from src.models.unet import UNet
from src.data.dataset import SyntheticMedicalDataset, get_data_loaders
from src.losses.medical_losses import get_loss_function
from src.metrics.segmentation_metrics import SegmentationMetrics
from src.utils.device import get_device, set_deterministic_seed
from src.utils.explainability import ExplainabilityAnalyzer


def main():
    """Main demonstration function."""
    print("🏥 Medical Image Segmentation Demo")
    print("=" * 50)
    print("⚠️  RESEARCH TOOL ONLY - NOT FOR CLINICAL USE")
    print("=" * 50)
    
    # Set deterministic seed for reproducibility
    set_deterministic_seed(42)
    print("✅ Set deterministic seed for reproducibility")
    
    # Get device
    device = get_device()
    print(f"✅ Using device: {device}")
    
    # Create synthetic dataset
    print("\n📊 Creating synthetic medical dataset...")
    dataset = SyntheticMedicalDataset(
        size=(256, 256),
        num_samples=100,
        num_structures=3,
        noise_level=0.1
    )
    
    # Create data loader
    loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)
    print(f"✅ Dataset created with {len(dataset)} samples")
    
    # Create model
    print("\n🧠 Creating U-Net model...")
    model = UNet(
        in_channels=1,
        out_channels=1,
        base_features=64,
        dropout_rate=0.1
    )
    model = model.to(device)
    
    # Print model info
    model_info = model.get_model_size()
    print(f"✅ Model created with {model_info['total_parameters']:,} parameters")
    print(f"   Model size: {model_info['model_size_mb']:.2f} MB")
    
    # Create loss function and optimizer
    print("\n⚙️  Setting up training...")
    criterion = get_loss_function("combined", dice_weight=0.5, bce_weight=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    
    # Training loop
    print("\n🚀 Starting training...")
    model.train()
    metrics = SegmentationMetrics()
    
    for epoch in range(5):  # Short demo training
        epoch_loss = 0.0
        epoch_metrics = SegmentationMetrics()
        
        for batch_idx, (images, masks) in enumerate(loader):
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
            with torch.no_grad():
                predictions = torch.sigmoid(outputs)
                epoch_metrics.update(predictions, masks)
        
        # Print epoch results
        avg_loss = epoch_loss / len(loader)
        epoch_results = epoch_metrics.compute()
        
        print(f"Epoch {epoch + 1}/5:")
        print(f"  Loss: {avg_loss:.4f}")
        if epoch_results:
            print(f"  Dice: {epoch_results.get('dice_mean', 0):.4f}")
            print(f"  IoU: {epoch_results.get('iou_mean', 0):.4f}")
    
    print("✅ Training completed!")
    
    # Evaluation
    print("\n📈 Evaluating model...")
    model.eval()
    evaluator_metrics = SegmentationMetrics()
    
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            predictions = torch.sigmoid(outputs)
            evaluator_metrics.update(predictions, masks)
    
    # Compute final metrics
    final_metrics = evaluator_metrics.compute()
    print("✅ Evaluation completed!")
    print("\n📊 Final Results:")
    print("-" * 30)
    for metric_name, value in final_metrics.items():
        print(f"{metric_name:20}: {value:.4f}")
    
    # Explainability analysis
    print("\n🔍 Generating explainability analysis...")
    try:
        analyzer = ExplainabilityAnalyzer(model)
        
        # Get a sample for analysis
        sample_image, sample_mask = dataset[0]
        sample_image = sample_image.unsqueeze(0).to(device)
        sample_mask = sample_mask.unsqueeze(0).to(device)
        
        # Analyze
        explain_results = analyzer.analyze(sample_image, sample_mask)
        
        print("✅ Explainability analysis completed!")
        
        # Visualize results
        print("\n🎨 Creating visualization...")
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original image
        axes[0, 0].imshow(sample_image.squeeze().cpu().numpy(), cmap='gray')
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Ground truth
        axes[0, 1].imshow(sample_mask.squeeze().cpu().numpy(), cmap='hot')
        axes[0, 1].set_title('Ground Truth')
        axes[0, 1].axis('off')
        
        # Prediction
        if explain_results.get("prediction") is not None:
            axes[0, 2].imshow(explain_results["prediction"], cmap='hot')
            axes[0, 2].set_title('Prediction')
            axes[0, 2].axis('off')
        
        # Grad-CAM
        if explain_results.get("gradcam") is not None:
            axes[1, 0].imshow(explain_results["gradcam"], cmap='jet')
            axes[1, 0].set_title('Grad-CAM')
            axes[1, 0].axis('off')
        
        # Uncertainty
        if explain_results.get("uncertainty") is not None:
            im = axes[1, 1].imshow(explain_results["uncertainty"], cmap='viridis')
            axes[1, 1].set_title('Uncertainty')
            axes[1, 1].axis('off')
            plt.colorbar(im, ax=axes[1, 1])
        
        # Hide unused subplot
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig('demo_results.png', dpi=300, bbox_inches='tight')
        print("✅ Visualization saved as 'demo_results.png'")
        
    except Exception as e:
        print(f"⚠️  Explainability analysis failed: {e}")
    
    # Summary
    print("\n" + "=" * 50)
    print("🎉 DEMO COMPLETED SUCCESSFULLY!")
    print("=" * 50)
    print("This demonstration showcased:")
    print("✅ Modern U-Net architecture with batch normalization")
    print("✅ Medical-specific loss functions (Dice + BCE)")
    print("✅ Comprehensive evaluation metrics")
    print("✅ Explainability analysis (Grad-CAM, uncertainty)")
    print("✅ Device-agnostic implementation")
    print("✅ Reproducible results with deterministic seeding")
    print("\n⚠️  REMEMBER: This is a research tool only!")
    print("   Do not use for clinical diagnosis or treatment.")


if __name__ == "__main__":
    main()
