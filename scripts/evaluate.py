#!/usr/bin/env python3
"""
Evaluation script for medical image segmentation models.

This script provides comprehensive evaluation of trained models including
metrics computation, visualization, and explainability analysis.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any

import yaml
import torch
from torch.utils.data import DataLoader

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.data.dataset import SyntheticMedicalDataset, get_data_loaders
from src.eval.evaluator import SegmentationEvaluator, load_model_for_evaluation
from src.utils.device import get_device


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate medical image segmentation model")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to test data directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda, mps, cpu)"
    )
    parser.add_argument(
        "--explainability",
        action="store_true",
        help="Enable explainability analysis"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Save prediction visualizations"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = get_device()
    
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from: {args.model}")
    model = load_model_for_evaluation(
        model_path=args.model,
        model_name=config['model']['name'],
        device=device
    )
    
    # Create test dataset
    if args.data:
        # For real medical data, implement dataset loading here
        print("Medical dataset evaluation not implemented in this demo")
        return
    else:
        # Use synthetic dataset for demonstration
        print("Creating synthetic test dataset...")
        test_dataset = SyntheticMedicalDataset(
            size=tuple(config['data']['synthetic']['image_size']),
            num_samples=200,  # Smaller test set
            num_structures=config['data']['synthetic']['num_structures'],
            noise_level=config['data']['synthetic']['noise_level'],
        )
    
    # Create test data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True,
    )
    
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Create evaluator
    evaluator = SegmentationEvaluator(
        model=model,
        device=device,
        save_dir=args.output,
    )
    
    # Run evaluation
    print("Running evaluation...")
    metrics = evaluator.evaluate_dataset(
        data_loader=test_loader,
        compute_explainability=args.explainability,
        save_predictions=args.visualize,
    )
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    
    for metric_name, value in metrics.items():
        print(f"{metric_name:20}: {value:.4f}")
    
    # Generate and save report
    report = evaluator.generate_report(metrics)
    print("\n" + report)
    
    # Save report to file
    report_path = Path(args.output) / "evaluation_report.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"\nEvaluation report saved to: {report_path}")
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
