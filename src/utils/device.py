"""Device management utilities for medical image segmentation."""

import os
import random
from typing import Optional

import numpy as np
import torch


def get_device() -> torch.device:
    """
    Get the best available device with fallback priority: CUDA -> MPS -> CPU.
    
    Returns:
        torch.device: The selected device.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name()}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon MPS device")
    else:
        device = torch.device("cpu")
        print("Using CPU device")
    
    return device


def set_deterministic_seed(seed: int = 42) -> None:
    """
    Set deterministic seed for reproducibility across all random number generators.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variables for additional reproducibility
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    
    # Enable deterministic algorithms where possible
    torch.use_deterministic_algorithms(True, warn_only=True)


def get_device_memory_info(device: Optional[torch.device] = None) -> dict:
    """
    Get memory information for the specified device.
    
    Args:
        device: Device to check. If None, uses current device.
        
    Returns:
        dict: Memory information including total and allocated memory.
    """
    if device is None:
        device = torch.cuda.current_device() if torch.cuda.is_available() else None
    
    if device is None or device.type == "cpu":
        return {"device": "cpu", "total_memory": "N/A", "allocated_memory": "N/A"}
    
    if device.type == "cuda":
        total_memory = torch.cuda.get_device_properties(device).total_memory
        allocated_memory = torch.cuda.memory_allocated(device)
        return {
            "device": str(device),
            "total_memory": f"{total_memory / 1024**3:.2f} GB",
            "allocated_memory": f"{allocated_memory / 1024**3:.2f} GB",
        }
    elif device.type == "mps":
        return {
            "device": str(device),
            "total_memory": "N/A (MPS)",
            "allocated_memory": "N/A (MPS)",
        }
    
    return {"device": str(device), "total_memory": "Unknown", "allocated_memory": "Unknown"}
