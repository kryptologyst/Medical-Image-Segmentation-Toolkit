"""Evaluation module for medical image segmentation."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.models.unet import UNet, UNetPlusPlus
from src.metrics.segmentation_metrics import SegmentationMetrics
from src.utils.device import get_device
from src.utils.explainability import ExplainabilityAnalyzer


class SegmentationEvaluator:
    """
    Evaluator class for medical image segmentation models.
    
    Provides comprehensive evaluation functionality including metrics computation,
    visualization, and explainability analysis.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
        save_dir: str = "evaluation_results",
    ) -> None:
        """
        Initialize evaluator.
        
        Args:
            model: The model to evaluate.
            device: Device to use for evaluation.
            save_dir: Directory to save evaluation results.
        """
        self.model = model
        self.device = device or get_device()
        self.save_dir = Path(save_dir)
        
        # Move model to device
        self.model.to(self.device)
        self.model.eval()
        
        # Create save directory
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize explainability analyzer
        self.explainability_analyzer = ExplainabilityAnalyzer(model)
    
    def evaluate_dataset(
        self,
        data_loader: DataLoader,
        compute_explainability: bool = False,
        save_predictions: bool = False,
    ) -> Dict[str, float]:
        """
        Evaluate model on a dataset.
        
        Args:
            data_loader: Data loader for evaluation.
            compute_explainability: Whether to compute explainability maps.
            save_predictions: Whether to save prediction visualizations.
            
        Returns:
            Dictionary containing evaluation metrics.
        """
        metrics = SegmentationMetrics()
        all_predictions = []
        all_targets = []
        explainability_results = []
        
        print("Evaluating model...")
        
        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(tqdm(data_loader)):
                # Move to device
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                predictions = torch.sigmoid(outputs)
                
                # Update metrics
                metrics.update(predictions, masks)
                
                # Store for additional analysis
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(masks.cpu().numpy())
                
                # Compute explainability for first few samples
                if compute_explainability and batch_idx < 5:
                    for i in range(min(images.size(0), 2)):  # Limit to 2 samples per batch
                        single_image = images[i:i+1]
                        single_mask = masks[i:i+1]
                        
                        try:
                            explain_results = self.explainability_analyzer.analyze(
                                single_image, single_mask
                            )
                            explainability_results.append({
                                'image': single_image.cpu().numpy(),
                                'mask': single_mask.cpu().numpy(),
                                'prediction': predictions[i].cpu().numpy(),
                                'explainability': explain_results,
                            })
                        except Exception as e:
                            print(f"Explainability analysis failed for sample {batch_idx}: {e}")
                
                # Save prediction visualizations
                if save_predictions and batch_idx < 10:  # Save first 10 batches
                    self._save_prediction_visualizations(
                        images.cpu().numpy(),
                        masks.cpu().numpy(),
                        predictions.cpu().numpy(),
                        batch_idx,
                    )
        
        # Compute final metrics
        final_metrics = metrics.compute()
        
        # Save explainability results
        if explainability_results:
            self._save_explainability_results(explainability_results)
        
        # Save evaluation summary
        self._save_evaluation_summary(final_metrics)
        
        return final_metrics
    
    def _save_prediction_visualizations(
        self,
        images: np.ndarray,
        masks: np.ndarray,
        predictions: np.ndarray,
        batch_idx: int,
    ) -> None:
        """Save prediction visualizations."""
        batch_size = images.shape[0]
        
        for i in range(min(batch_size, 4)):  # Save max 4 samples per batch
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Original image
            axes[0].imshow(images[i, 0], cmap='gray')
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            # Ground truth
            axes[1].imshow(masks[i, 0], cmap='hot')
            axes[1].set_title('Ground Truth')
            axes[1].axis('off')
            
            # Prediction
            axes[2].imshow(predictions[i, 0], cmap='hot')
            axes[2].set_title('Prediction')
            axes[2].axis('off')
            
            plt.tight_layout()
            plt.savefig(
                self.save_dir / f"prediction_batch_{batch_idx}_sample_{i}.png",
                dpi=300,
                bbox_inches='tight',
            )
            plt.close()
    
    def _save_explainability_results(self, results: List[Dict]) -> None:
        """Save explainability analysis results."""
        explainability_dir = self.save_dir / "explainability"
        explainability_dir.mkdir(exist_ok=True)
        
        for i, result in enumerate(results):
            try:
                self.explainability_analyzer.visualize_results(
                    result['image'],
                    result['explainability'],
                    result['mask'],
                    save_path=str(explainability_dir / f"explainability_sample_{i}.png"),
                )
            except Exception as e:
                print(f"Failed to save explainability visualization {i}: {e}")
    
    def _save_evaluation_summary(self, metrics: Dict[str, float]) -> None:
        """Save evaluation summary to file."""
        summary_path = self.save_dir / "evaluation_summary.txt"
        
        with open(summary_path, 'w') as f:
            f.write("Medical Image Segmentation Evaluation Summary\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("Segmentation Metrics:\n")
            f.write("-" * 20 + "\n")
            
            for metric_name, value in metrics.items():
                f.write(f"{metric_name}: {value:.4f}\n")
            
            f.write("\nInterpretation:\n")
            f.write("-" * 15 + "\n")
            f.write("Dice Score: Higher is better (0-1, 1 = perfect)\n")
            f.write("IoU Score: Higher is better (0-1, 1 = perfect)\n")
            f.write("Sensitivity: Higher is better (0-1, 1 = perfect)\n")
            f.write("Specificity: Higher is better (0-1, 1 = perfect)\n")
            f.write("Hausdorff Distance: Lower is better (0-inf)\n")
            f.write("Average Surface Distance: Lower is better (0-inf)\n")
        
        print(f"Evaluation summary saved to {summary_path}")
    
    def compare_models(
        self,
        models: Dict[str, nn.Module],
        data_loader: DataLoader,
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare multiple models on the same dataset.
        
        Args:
            models: Dictionary of model names and model instances.
            data_loader: Data loader for evaluation.
            
        Returns:
            Dictionary containing metrics for each model.
        """
        results = {}
        
        for model_name, model in models.items():
            print(f"Evaluating {model_name}...")
            
            # Create evaluator for this model
            evaluator = SegmentationEvaluator(model, self.device, self.save_dir)
            
            # Evaluate model
            metrics = evaluator.evaluate_dataset(data_loader)
            results[model_name] = metrics
        
        # Save comparison results
        self._save_model_comparison(results)
        
        return results
    
    def _save_model_comparison(self, results: Dict[str, Dict[str, float]]) -> None:
        """Save model comparison results."""
        comparison_path = self.save_dir / "model_comparison.txt"
        
        with open(comparison_path, 'w') as f:
            f.write("Model Comparison Results\n")
            f.write("=" * 25 + "\n\n")
            
            # Get all metric names
            all_metrics = set()
            for model_results in results.values():
                all_metrics.update(model_results.keys())
            
            # Write header
            f.write("Model".ljust(20))
            for metric in sorted(all_metrics):
                f.write(metric.ljust(15))
            f.write("\n" + "-" * (20 + 15 * len(all_metrics)) + "\n")
            
            # Write results
            for model_name, metrics in results.items():
                f.write(model_name.ljust(20))
                for metric in sorted(all_metrics):
                    value = metrics.get(metric, 0.0)
                    f.write(f"{value:.4f}".ljust(15))
                f.write("\n")
        
        print(f"Model comparison saved to {comparison_path}")
    
    def generate_report(self, metrics: Dict[str, float]) -> str:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            metrics: Dictionary containing evaluation metrics.
            
        Returns:
            Formatted report string.
        """
        report = []
        report.append("Medical Image Segmentation Evaluation Report")
        report.append("=" * 50)
        report.append("")
        
        # Overall performance
        report.append("Overall Performance:")
        report.append("-" * 20)
        
        dice_score = metrics.get('dice_mean', 0.0)
        iou_score = metrics.get('iou_mean', 0.0)
        
        report.append(f"Dice Score: {dice_score:.4f}")
        report.append(f"IoU Score: {iou_score:.4f}")
        report.append("")
        
        # Detailed metrics
        report.append("Detailed Metrics:")
        report.append("-" * 15)
        
        for metric_name, value in sorted(metrics.items()):
            report.append(f"{metric_name}: {value:.4f}")
        
        report.append("")
        
        # Performance interpretation
        report.append("Performance Interpretation:")
        report.append("-" * 25)
        
        if dice_score >= 0.9:
            report.append("Excellent segmentation performance (Dice ≥ 0.9)")
        elif dice_score >= 0.8:
            report.append("Good segmentation performance (Dice ≥ 0.8)")
        elif dice_score >= 0.7:
            report.append("Moderate segmentation performance (Dice ≥ 0.7)")
        elif dice_score >= 0.6:
            report.append("Fair segmentation performance (Dice ≥ 0.6)")
        else:
            report.append("Poor segmentation performance (Dice < 0.6)")
        
        return "\n".join(report)


def load_model_for_evaluation(
    model_path: str,
    model_name: str = "unet",
    device: Optional[torch.device] = None,
) -> nn.Module:
    """
    Load a trained model for evaluation.
    
    Args:
        model_path: Path to the model checkpoint.
        model_name: Name of the model architecture.
        device: Device to load the model on.
        
    Returns:
        Loaded model instance.
    """
    device = device or get_device()
    
    # Create model
    if model_name.lower() == 'unet':
        model = UNet()
    elif model_name.lower() == 'unet++':
        model = UNetPlusPlus()
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model
