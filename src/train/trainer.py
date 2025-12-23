"""Training module for medical image segmentation."""

import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from src.models.unet import UNet, UNetPlusPlus
from src.losses.medical_losses import get_loss_function
from src.metrics.segmentation_metrics import SegmentationMetrics
from src.utils.device import get_device, set_deterministic_seed


class SegmentationTrainer:
    """
    Trainer class for medical image segmentation models.
    
    Provides comprehensive training functionality including validation,
    checkpointing, and metrics tracking.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        loss_function: str = "combined",
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: Optional[torch.device] = None,
        save_dir: str = "checkpoints",
        log_interval: int = 10,
    ) -> None:
        """
        Initialize trainer.
        
        Args:
            model: The model to train.
            train_loader: Training data loader.
            val_loader: Validation data loader (optional).
            loss_function: Name of loss function to use.
            optimizer: Optimizer (if None, uses Adam).
            scheduler: Learning rate scheduler (optional).
            device: Device to use for training.
            save_dir: Directory to save checkpoints.
            log_interval: Interval for logging training progress.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device or get_device()
        self.save_dir = Path(save_dir)
        self.log_interval = log_interval
        
        # Move model to device
        self.model.to(self.device)
        
        # Setup loss function
        self.criterion = get_loss_function(loss_function)
        
        # Setup optimizer
        if optimizer is None:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=1e-3,
                weight_decay=1e-5,
            )
        else:
            self.optimizer = optimizer
        
        # Setup scheduler
        self.scheduler = scheduler
        
        # Create save directory
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.current_epoch = 0
        self.best_dice = 0.0
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = []
        self.val_metrics = []
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary containing training metrics.
        """
        self.model.train()
        total_loss = 0.0
        metrics = SegmentationMetrics()
        
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch + 1}",
            leave=False,
        )
        
        for batch_idx, (images, masks) in enumerate(progress_bar):
            # Move to device
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            metrics.update(outputs, masks)
            
            # Update progress bar
            if batch_idx % self.log_interval == 0:
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Avg Loss': f'{total_loss / (batch_idx + 1):.4f}',
                })
        
        # Compute epoch metrics
        epoch_loss = total_loss / len(self.train_loader)
        epoch_metrics = metrics.compute()
        
        return {
            'loss': epoch_loss,
            **epoch_metrics,
        }
    
    def validate(self) -> Dict[str, float]:
        """
        Validate the model.
        
        Returns:
            Dictionary containing validation metrics.
        """
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        total_loss = 0.0
        metrics = SegmentationMetrics()
        
        with torch.no_grad():
            for images, masks in tqdm(self.val_loader, desc="Validation", leave=False):
                # Move to device
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                
                # Update metrics
                total_loss += loss.item()
                metrics.update(outputs, masks)
        
        # Compute validation metrics
        val_loss = total_loss / len(self.val_loader)
        val_metrics = metrics.compute()
        
        return {
            'loss': val_loss,
            **val_metrics,
        }
    
    def train(
        self,
        num_epochs: int,
        save_best: bool = True,
        save_last: bool = True,
        early_stopping_patience: Optional[int] = None,
    ) -> Dict[str, list]:
        """
        Train the model for multiple epochs.
        
        Args:
            num_epochs: Number of epochs to train.
            save_best: Whether to save best model based on validation Dice.
            save_last: Whether to save last model.
            early_stopping_patience: Patience for early stopping (None to disable).
            
        Returns:
            Dictionary containing training history.
        """
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        best_epoch = 0
        patience_counter = 0
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Training
            train_metrics = self.train_epoch()
            self.train_losses.append(train_metrics['loss'])
            self.train_metrics.append(train_metrics)
            
            # Validation
            val_metrics = self.validate()
            if val_metrics:
                self.val_losses.append(val_metrics['loss'])
                self.val_metrics.append(val_metrics)
            
            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics.get('loss', train_metrics['loss']))
                else:
                    self.scheduler.step()
            
            # Logging
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"Train Loss: {train_metrics['loss']:.4f}")
            if val_metrics:
                print(f"Val Loss: {val_metrics['loss']:.4f}")
                if 'dice_mean' in val_metrics:
                    print(f"Val Dice: {val_metrics['dice_mean']:.4f}")
            
            # Save checkpoints
            if save_best and val_metrics and 'dice_mean' in val_metrics:
                if val_metrics['dice_mean'] > self.best_dice:
                    self.best_dice = val_metrics['dice_mean']
                    best_epoch = epoch
                    self.save_checkpoint('best_model.pth')
                    print(f"New best model saved (Dice: {self.best_dice:.4f})")
            
            if save_last:
                self.save_checkpoint('last_model.pth')
            
            # Early stopping
            if early_stopping_patience is not None and val_metrics:
                if 'dice_mean' in val_metrics:
                    if val_metrics['dice_mean'] <= self.best_dice:
                        patience_counter += 1
                    else:
                        patience_counter = 0
                    
                    if patience_counter >= early_stopping_patience:
                        print(f"Early stopping at epoch {epoch + 1}")
                        break
        
        print(f"Training completed. Best Dice: {self.best_dice:.4f} at epoch {best_epoch + 1}")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics,
        }
    
    def save_checkpoint(self, filename: str) -> None:
        """
        Save model checkpoint.
        
        Args:
            filename: Name of the checkpoint file.
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_dice': self.best_dice,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics,
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, self.save_dir / filename)
    
    def load_checkpoint(self, filename: str) -> None:
        """
        Load model checkpoint.
        
        Args:
            filename: Name of the checkpoint file.
        """
        checkpoint_path = self.save_dir / filename
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_dice = checkpoint['best_dice']
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.train_metrics = checkpoint.get('train_metrics', [])
        self.val_metrics = checkpoint.get('val_metrics', [])
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"Checkpoint loaded from {checkpoint_path}")
        print(f"Epoch: {self.current_epoch}, Best Dice: {self.best_dice:.4f}")


def create_trainer(
    model_name: str,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    config: Optional[Dict] = None,
) -> SegmentationTrainer:
    """
    Create a trainer with the specified model and configuration.
    
    Args:
        model_name: Name of the model ('unet', 'unet++').
        train_loader: Training data loader.
        val_loader: Validation data loader (optional).
        config: Configuration dictionary (optional).
        
    Returns:
        Configured trainer instance.
    """
    if config is None:
        config = {}
    
    # Create model
    if model_name.lower() == 'unet':
        model = UNet(
            in_channels=config.get('in_channels', 1),
            out_channels=config.get('out_channels', 1),
            base_features=config.get('base_features', 64),
            dropout_rate=config.get('dropout_rate', 0.1),
        )
    elif model_name.lower() == 'unet++':
        model = UNetPlusPlus(
            in_channels=config.get('in_channels', 1),
            out_channels=config.get('out_channels', 1),
            base_features=config.get('base_features', 64),
            dropout_rate=config.get('dropout_rate', 0.1),
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Create trainer
    trainer = SegmentationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_function=config.get('loss_function', 'combined'),
        device=config.get('device'),
        save_dir=config.get('save_dir', 'checkpoints'),
        log_interval=config.get('log_interval', 10),
    )
    
    return trainer
