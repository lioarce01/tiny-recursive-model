"""
Training utilities for TRM: EMA, checkpointing, metrics.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import json


class EMA:
    """Exponential Moving Average for model weights."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        # Initialize shadow weights
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model: nn.Module):
        """Update EMA weights."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        """Apply EMA weights to model."""
        self._backup()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.shadow[name]

    def restore(self):
        """Restore original model weights."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}

    def _backup(self):
        """Backup current model weights."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    filepath: Path,
    config: Dict[str, Any]
):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'config': config
    }

    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(
    filepath: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None
) -> Tuple[int, Dict[str, float], Dict[str, Any]]:
    """Load model checkpoint."""
    checkpoint = torch.load(filepath)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    epoch = checkpoint['epoch']
    metrics = checkpoint.get('metrics', {})
    config = checkpoint.get('config', {})

    print(f"Checkpoint loaded from {filepath} (epoch {epoch})")
    return epoch, metrics, config


def compute_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute accuracy for classification tasks."""
    _, predicted = torch.max(predictions, dim=1)
    correct = (predicted == targets).sum().item()
    total = targets.size(0)
    return correct / total


def compute_sequence_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute sequence-level accuracy (all tokens correct)."""
    _, predicted = torch.max(predictions, dim=-1)  # (batch_size, seq_len)
    correct = (predicted == targets).all(dim=-1).sum().item()
    total = targets.size(0)
    return correct / total


def get_model_size_mb(model: nn.Module) -> float:
    """Get model size in MB."""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def setup_logging(log_dir: Path, experiment_name: str) -> None:
    """Setup logging directory and files."""
    log_dir.mkdir(exist_ok=True)

    # Could integrate with wandb here if needed
    # import wandb
    # wandb.init(project="trm", name=experiment_name)

    print(f"Logging to {log_dir}")


def log_metrics(
    metrics: Dict[str, float],
    step: int,
    log_dir: Path,
    prefix: str = ""
) -> None:
    """Log metrics to console and optionally to wandb/tensorboard."""
    log_str = f"Step {step}: "
    for key, value in metrics.items():
        log_str += f"{prefix}{key}: {value:.4f}, "

    print(log_str.rstrip(", "))

    # Could log to wandb/tensorboard here
    # wandb.log({f"{prefix}{k}": v for k, v in metrics.items()}, step=step)


def create_experiment_dir(base_dir: str, experiment_name: str) -> Path:
    """Create experiment directory with timestamp."""
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(base_dir) / f"{experiment_name}_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


def save_config(config: Dict[str, Any], filepath: Path) -> None:
    """Save configuration to JSON file."""
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2)


def load_config(filepath: Path) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)