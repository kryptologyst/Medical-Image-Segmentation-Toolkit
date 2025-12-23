#!/usr/bin/env python3
"""
Main training script for medical image segmentation.

This script provides a complete training pipeline for medical image segmentation
using U-Net and UNet++ architectures with various loss functions and evaluation metrics.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Any

import yaml
import torch
from torch.utils.data import DataLoader

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.data.dataset import SyntheticMedicalDataset, get_data_loaders
from src.train.trainer import create_trainer
from src.utils.device import get_device, set_deterministic_seed


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file.
        
    Returns:
        Configuration dictionary.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_datasets(config: Dict[str, Any]) -> tuple:
    """
    Create training, validation, and test datasets.
    
    Args:
        config: Configuration dictionary.
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset).
    """
    data_config = config['data']
    
    if data_config['dataset_type'] == 'synthetic':
        # Create synthetic datasets
        synthetic_config = data_config['synthetic']
        
        # Full dataset
        full_dataset = SyntheticMedicalDataset(
            size=tuple(synthetic_config['image_size']),
            num_samples=synthetic_config['num_samples'],
            num_structures=synthetic_config['num_structures'],
            noise_level=synthetic_config['noise_level'],
        )
        
        # Split dataset
        train_size = int(len(full_dataset) * data_config['train_split'])
        val_size = int(len(full_dataset) * data_config['val_split'])
        test_size = len(full_dataset) - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size, test_size]
        )
        
    else:
        # For medical datasets, you would implement the actual dataset loading here
        raise NotImplementedError("Medical dataset loading not implemented in this demo")
    
    return train_dataset, val_dataset, test_dataset


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train medical image segmentation model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda, mps, cpu)"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    elif config['device']['auto_detect']:
        device = get_device()
    else:
        device = torch.device(config['device']['manual_device'])
    
    print(f"Using device: {device}")
    
    # Set random seed for reproducibility
    if config['deterministic']:
        set_deterministic_seed(config['seed'])
        print(f"Set deterministic seed: {config['seed']}")
    
    # Create datasets
    print("Creating datasets...")
    train_dataset, val_dataset, test_dataset = create_datasets(config)
    
    print(f"Dataset sizes:")
    print(f"  Train: {len(train_dataset)}")
    print(f"  Validation: {len(val_dataset)}")
    print(f"  Test: {len(test_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=config['data']['shuffle_train'],
        num_workers=config['data']['num_workers'],
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True,
    )
    
    # Create trainer
    print("Creating trainer...")
    trainer = create_trainer(
        model_name=config['model']['name'],
        train_loader=train_loader,
        val_loader=val_loader,
        config={
            'in_channels': config['model']['in_channels'],
            'out_channels': config['model']['out_channels'],
            'base_features': config['model']['base_features'],
            'dropout_rate': config['model']['dropout_rate'],
            'loss_function': config['training']['loss_function'],
            'device': device,
            'save_dir': config['training']['checkpoint_dir'],
            'log_interval': config['logging']['log_interval'],
        }
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Train model
    print("Starting training...")
    history = trainer.train(
        num_epochs=config['training']['num_epochs'],
        save_best=config['training']['save_best'],
        save_last=config['training']['save_last'],
        early_stopping_patience=config['training']['early_stopping_patience'],
    )
    
    print("Training completed!")
    
    # Evaluate on test set
    print("Evaluating on test set...")
    from src.eval.evaluator import SegmentationEvaluator
    
    evaluator = SegmentationEvaluator(
        model=trainer.model,
        device=device,
        save_dir=config['evaluation']['save_dir'],
    )
    
    test_metrics = evaluator.evaluate_dataset(
        data_loader=test_loader,
        compute_explainability=config['evaluation']['compute_explainability'],
        save_predictions=config['evaluation']['save_predictions'],
    )
    
    print("Test set evaluation completed!")
    print("Test metrics:")
    for metric_name, value in test_metrics.items():
        print(f"  {metric_name}: {value:.4f}")
    
    # Generate report
    report = evaluator.generate_report(test_metrics)
    print("\n" + "="*50)
    print(report)
    
    # Save final report
    report_path = Path(config['evaluation']['save_dir']) / "final_report.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"\nFinal report saved to: {report_path}")


if __name__ == "__main__":
    main()
