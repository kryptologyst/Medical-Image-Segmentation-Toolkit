"""Explainability and uncertainty estimation for medical image segmentation."""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import cv2


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM) for medical image segmentation.
    
    Grad-CAM generates visual explanations for decisions made by CNN-based models
    by highlighting important regions in the input image.
    """
    
    def __init__(self, model: nn.Module, target_layer: str) -> None:
        """
        Initialize Grad-CAM.
        
        Args:
            model: The model to explain.
            target_layer: Name of the target layer for Grad-CAM.
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hooks = []
        
        self._register_hooks()
    
    def _register_hooks(self) -> None:
        """Register forward and backward hooks."""
        def forward_hook(module, input, output):
            self.activations = output
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        # Find the target layer
        target_module = None
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                target_module = module
                break
        
        if target_module is None:
            raise ValueError(f"Target layer '{self.target_layer}' not found in model")
        
        # Register hooks
        self.hooks.append(target_module.register_forward_hook(forward_hook))
        self.hooks.append(target_module.register_backward_hook(backward_hook))
    
    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        class_idx: Optional[int] = None,
        retain_graph: bool = False,
    ) -> np.ndarray:
        """
        Generate Grad-CAM for the given input.
        
        Args:
            input_tensor: Input tensor of shape (1, C, H, W).
            class_idx: Class index for which to generate CAM (None for binary).
            retain_graph: Whether to retain computational graph.
            
        Returns:
            Grad-CAM heatmap as numpy array.
        """
        # Forward pass
        self.model.eval()
        output = self.model(input_tensor)
        
        # For binary segmentation, use the output directly
        if class_idx is None:
            if output.shape[1] == 1:  # Binary segmentation
                target = output
            else:
                target = output[:, 0:1]  # Use first channel
        else:
            target = output[:, class_idx:class_idx+1]
        
        # Backward pass
        self.model.zero_grad()
        target.backward(retain_graph=retain_graph)
        
        # Generate CAM
        gradients = self.gradients[0]  # Remove batch dimension
        activations = self.activations[0]  # Remove batch dimension
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(1, 2), keepdim=True)
        
        # Weighted combination of activation maps
        cam = torch.sum(weights * activations, dim=0, keepdim=True)
        cam = F.relu(cam)
        
        # Normalize to [0, 1]
        cam = cam - cam.min()
        cam = cam / cam.max()
        
        # Convert to numpy
        cam_np = cam.squeeze().cpu().detach().numpy()
        
        return cam_np
    
    def __del__(self):
        """Remove hooks when object is deleted."""
        for hook in self.hooks:
            hook.remove()


class UncertaintyEstimator:
    """
    Uncertainty estimation for medical image segmentation using Monte Carlo Dropout.
    
    This class provides uncertainty estimates by running multiple forward passes
    with dropout enabled during inference.
    """
    
    def __init__(self, model: nn.Module, num_samples: int = 10) -> None:
        """
        Initialize uncertainty estimator.
        
        Args:
            model: The model for uncertainty estimation.
            num_samples: Number of Monte Carlo samples.
        """
        self.model = model
        self.num_samples = num_samples
    
    def estimate_uncertainty(
        self,
        input_tensor: torch.Tensor,
        enable_dropout: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate uncertainty using Monte Carlo Dropout.
        
        Args:
            input_tensor: Input tensor of shape (1, C, H, W).
            enable_dropout: Whether to enable dropout during inference.
            
        Returns:
            Tuple of (mean_prediction, uncertainty_map).
        """
        predictions = []
        
        # Enable dropout if requested
        if enable_dropout:
            self._enable_dropout()
        
        # Monte Carlo sampling
        with torch.no_grad():
            for _ in range(self.num_samples):
                pred = self.model(input_tensor)
                pred = torch.sigmoid(pred)  # Convert to probabilities
                predictions.append(pred.cpu().numpy())
        
        # Disable dropout
        if enable_dropout:
            self._disable_dropout()
        
        # Convert to numpy array
        predictions = np.array(predictions)  # Shape: (num_samples, 1, H, W)
        
        # Compute mean and uncertainty
        mean_pred = np.mean(predictions, axis=0).squeeze()
        uncertainty = np.std(predictions, axis=0).squeeze()
        
        return mean_pred, uncertainty
    
    def _enable_dropout(self) -> None:
        """Enable dropout layers for uncertainty estimation."""
        for module in self.model.modules():
            if isinstance(module, nn.Dropout) or isinstance(module, nn.Dropout2d):
                module.train()
    
    def _disable_dropout(self) -> None:
        """Disable dropout layers."""
        for module in self.model.modules():
            if isinstance(module, nn.Dropout) or isinstance(module, nn.Dropout2d):
                module.eval()


class AttentionVisualizer:
    """
    Visualize attention maps from transformer-based models.
    
    This class extracts and visualizes attention weights from transformer
    layers in models like Vision Transformers or Swin Transformers.
    """
    
    def __init__(self, model: nn.Module) -> None:
        """
        Initialize attention visualizer.
        
        Args:
            model: The model containing attention layers.
        """
        self.model = model
        self.attention_maps = {}
        self.hooks = []
        
        self._register_attention_hooks()
    
    def _register_attention_hooks(self) -> None:
        """Register hooks to capture attention weights."""
        def attention_hook(module, input, output):
            # Extract attention weights from transformer layers
            if hasattr(module, 'attention_weights'):
                self.attention_maps[id(module)] = module.attention_weights
        
        # Register hooks for attention layers
        for name, module in self.model.named_modules():
            if 'attention' in name.lower() or 'attn' in name.lower():
                self.hooks.append(module.register_forward_hook(attention_hook))
    
    def visualize_attention(
        self,
        input_tensor: torch.Tensor,
        layer_name: Optional[str] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Visualize attention maps.
        
        Args:
            input_tensor: Input tensor.
            layer_name: Specific layer to visualize (None for all).
            
        Returns:
            Dictionary of attention maps.
        """
        # Forward pass to capture attention
        with torch.no_grad():
            _ = self.model(input_tensor)
        
        # Extract attention maps
        attention_maps = {}
        for module_id, attention in self.attention_maps.items():
            if attention is not None:
                # Convert to numpy and process
                attn_np = attention.cpu().numpy()
                attention_maps[f"layer_{module_id}"] = attn_np
        
        return attention_maps
    
    def __del__(self):
        """Remove hooks when object is deleted."""
        for hook in self.hooks:
            hook.remove()


class ExplainabilityAnalyzer:
    """
    Comprehensive explainability analysis for medical image segmentation.
    
    This class combines multiple explainability techniques to provide
    comprehensive insights into model decisions.
    """
    
    def __init__(
        self,
        model: nn.Module,
        target_layer: str = "bottleneck",
        num_mc_samples: int = 10,
    ) -> None:
        """
        Initialize explainability analyzer.
        
        Args:
            model: The model to analyze.
            target_layer: Target layer for Grad-CAM.
            num_mc_samples: Number of Monte Carlo samples for uncertainty.
        """
        self.model = model
        self.gradcam = GradCAM(model, target_layer)
        self.uncertainty_estimator = UncertaintyEstimator(model, num_mc_samples)
        self.attention_visualizer = AttentionVisualizer(model)
    
    def analyze(
        self,
        input_tensor: torch.Tensor,
        ground_truth: Optional[torch.Tensor] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Perform comprehensive explainability analysis.
        
        Args:
            input_tensor: Input tensor of shape (1, C, H, W).
            ground_truth: Ground truth mask (optional).
            
        Returns:
            Dictionary containing various explainability maps.
        """
        results = {}
        
        # Generate Grad-CAM
        try:
            gradcam = self.gradcam.generate_cam(input_tensor)
            results["gradcam"] = gradcam
        except Exception as e:
            print(f"Grad-CAM failed: {e}")
            results["gradcam"] = None
        
        # Estimate uncertainty
        try:
            mean_pred, uncertainty = self.uncertainty_estimator.estimate_uncertainty(input_tensor)
            results["mean_prediction"] = mean_pred
            results["uncertainty"] = uncertainty
        except Exception as e:
            print(f"Uncertainty estimation failed: {e}")
            results["mean_prediction"] = None
            results["uncertainty"] = None
        
        # Visualize attention (if available)
        try:
            attention_maps = self.attention_visualizer.visualize_attention(input_tensor)
            results["attention_maps"] = attention_maps
        except Exception as e:
            print(f"Attention visualization failed: {e}")
            results["attention_maps"] = None
        
        # Generate prediction
        with torch.no_grad():
            prediction = self.model(input_tensor)
            prediction = torch.sigmoid(prediction)
            results["prediction"] = prediction.squeeze().cpu().numpy()
        
        return results
    
    def visualize_results(
        self,
        input_image: np.ndarray,
        results: Dict[str, np.ndarray],
        ground_truth: Optional[np.ndarray] = None,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Visualize explainability results.
        
        Args:
            input_image: Original input image.
            results: Results from analyze() method.
            ground_truth: Ground truth mask (optional).
            save_path: Path to save visualization (optional).
        """
        num_plots = 3
        if ground_truth is not None:
            num_plots += 1
        if results.get("uncertainty") is not None:
            num_plots += 1
        
        fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5))
        if num_plots == 1:
            axes = [axes]
        
        plot_idx = 0
        
        # Original image
        axes[plot_idx].imshow(input_image.squeeze(), cmap='gray')
        axes[plot_idx].set_title('Original Image')
        axes[plot_idx].axis('off')
        plot_idx += 1
        
        # Prediction
        if results.get("prediction") is not None:
            axes[plot_idx].imshow(results["prediction"], cmap='hot')
            axes[plot_idx].set_title('Prediction')
            axes[plot_idx].axis('off')
            plot_idx += 1
        
        # Ground truth
        if ground_truth is not None:
            axes[plot_idx].imshow(ground_truth.squeeze(), cmap='hot')
            axes[plot_idx].set_title('Ground Truth')
            axes[plot_idx].axis('off')
            plot_idx += 1
        
        # Grad-CAM
        if results.get("gradcam") is not None:
            axes[plot_idx].imshow(results["gradcam"], cmap='jet')
            axes[plot_idx].set_title('Grad-CAM')
            axes[plot_idx].axis('off')
            plot_idx += 1
        
        # Uncertainty
        if results.get("uncertainty") is not None:
            im = axes[plot_idx].imshow(results["uncertainty"], cmap='viridis')
            axes[plot_idx].set_title('Uncertainty')
            axes[plot_idx].axis('off')
            plt.colorbar(im, ax=axes[plot_idx])
            plot_idx += 1
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def __del__(self):
        """Clean up resources."""
        if hasattr(self, 'gradcam'):
            del self.gradcam
        if hasattr(self, 'attention_visualizer'):
            del self.attention_visualizer
