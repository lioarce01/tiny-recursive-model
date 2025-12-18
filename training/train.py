"""
TRM training script with AdamW, warmup, EMA, and deep supervision.

Supports training on ARC-AGI, Sudoku-Extreme, and Maze-Hard datasets.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR
import os
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import time
import logging

from models.trm import TinyRecursiveModel, create_trm_model
from data.datasets import create_data_loader, get_vocab_size
from training.utils import EMA, save_checkpoint, load_checkpoint, compute_accuracy


class WarmupScheduler:
    """Linear warmup scheduler."""

    def __init__(self, optimizer: optim.Optimizer, warmup_steps: int, base_lr: float):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.base_lr = base_lr
        self.current_step = 0

    def step(self):
        self.current_step += 1
        if self.current_step <= self.warmup_steps:
            lr = self.base_lr * (self.current_step / self.warmup_steps)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

    def get_lr(self) -> float:
        if self.current_step <= self.warmup_steps:
            return self.base_lr * (self.current_step / self.warmup_steps)
        return self.base_lr


def train_epoch(
    model: TinyRecursiveModel,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler: WarmupScheduler,
    ema_model: Optional[EMA],
    device: torch.device,
    epoch: int,
    config: Dict[str, Any],
    logger: logging.Logger
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_accuracy = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(train_loader):
        # Move batch to device
        input_tokens = batch['input_tokens'].to(device)
        input_mask = batch['input_mask'].to(device)
        target_tokens = batch['output_tokens'].to(device)

        # For simplicity, use first token of target as classification target
        # In practice, you'd handle sequence-to-sequence tasks differently
        targets = target_tokens[:, 0].long()

        # Forward pass with deep supervision
        optimizer.zero_grad()

        # Accumulate gradients for gradient_accumulation_steps
        loss = 0.0
        for accum_step in range(config['gradient_accumulation_steps']):
            start_idx = accum_step * (input_tokens.size(0) // config['gradient_accumulation_steps'])
            end_idx = (accum_step + 1) * (input_tokens.size(0) // config['gradient_accumulation_steps'])

            batch_input = input_tokens[start_idx:end_idx]
            batch_targets = targets[start_idx:end_idx]

            batch_loss = model.get_loss(
                batch_input,
                batch_targets,
                K=config['recursion_depth'],
                supervision_weight=config['supervision_weight']
            )
            loss += batch_loss / config['gradient_accumulation_steps']

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])

        # Optimizer step
        optimizer.step()

        # Update EMA model
        if ema_model is not None:
            ema_model.update(model)

        # Update learning rate
        scheduler.step()

        # Compute accuracy (simplified - in practice more sophisticated metrics)
        with torch.no_grad():
            final_pred, _ = model(batch_input, K=config['recursion_depth'], return_all_steps=False)
            accuracy = compute_accuracy(final_pred, batch_targets)

        total_loss += loss.item()
        total_accuracy += accuracy
        num_batches += 1

        # Log progress
        if batch_idx % config['log_interval'] == 0:
            current_lr = scheduler.get_lr()
            logger.info(
                f"Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, "
                f"Loss: {loss.item():.4f}, Acc: {accuracy:.4f}, LR: {current_lr:.6f}"
            )

    return {
        'train_loss': total_loss / num_batches,
        'train_accuracy': total_accuracy / num_batches
    }


def validate(
    model: TinyRecursiveModel,
    val_loader: DataLoader,
    device: torch.device,
    config: Dict[str, Any]
) -> Dict[str, float]:
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    total_accuracy = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            input_tokens = batch['input_tokens'].to(device)
            input_mask = batch['input_mask'].to(device)
            target_tokens = batch['output_tokens'].to(device)
            targets = target_tokens[:, 0].long()

            loss = model.get_loss(
                input_tokens,
                targets,
                K=config['recursion_depth'],
                supervision_weight=config['supervision_weight']
            )

            final_pred, _ = model(input_tokens, K=config['recursion_depth'], return_all_steps=False)
            accuracy = compute_accuracy(final_pred, targets)

            total_loss += loss.item()
            total_accuracy += accuracy
            num_batches += 1

    return {
        'val_loss': total_loss / num_batches,
        'val_accuracy': total_accuracy / num_batches
    }


def train_model(config: Dict[str, Any]) -> None:
    """Main training function."""
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Set random seeds
    torch.manual_seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['seed'])

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Create data loaders
    train_loader = create_data_loader(
        config['dataset'],
        config['data_path'],
        split='train',
        batch_size=config['batch_size'],
        max_samples=config['max_train_samples']
    )

    val_loader = create_data_loader(
        config['dataset'],
        config['data_path'],
        split='val',
        batch_size=config['batch_size'],
        shuffle=False,
        max_samples=config['max_val_samples']
    )

    # Create model
    vocab_size = get_vocab_size(config['dataset'])
    model = create_trm_model(
        vocab_size=vocab_size,
        max_seq_len=config['max_seq_len'],
        embed_dim=config['embed_dim'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        ff_dim=config['ff_dim'],
        latent_dim=config['latent_dim']
    ).to(device)

    logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
        betas=(0.9, 0.999)
    )

    # Create warmup scheduler
    scheduler = WarmupScheduler(optimizer, config['warmup_steps'], config['learning_rate'])

    # Create EMA model
    ema_model = EMA(model, decay=config['ema_decay']) if config['use_ema'] else None

    # Create output directory
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(exist_ok=True)

    # Training loop
    best_val_accuracy = 0.0
    patience_counter = 0

    for epoch in range(config['num_epochs']):
        logger.info(f"Starting epoch {epoch + 1}/{config['num_epochs']}")

        # Train epoch
        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, ema_model,
            device, epoch, config, logger
        )

        # Validate
        val_metrics = validate(model, val_loader, device, config)

        logger.info(
            f"Epoch {epoch + 1} - Train Loss: {train_metrics['train_loss']:.4f}, "
            f"Train Acc: {train_metrics['train_accuracy']:.4f}, "
            f"Val Loss: {val_metrics['val_loss']:.4f}, "
            f"Val Acc: {val_metrics['val_accuracy']:.4f}"
        )

        # Save best model
        if val_metrics['val_accuracy'] > best_val_accuracy:
            best_val_accuracy = val_metrics['val_accuracy']
            patience_counter = 0

            # Save regular model
            save_checkpoint(
                model, optimizer, epoch, val_metrics,
                output_dir / 'best_model.pth', config
            )

            # Save EMA model if available
            if ema_model is not None:
                save_checkpoint(
                    ema_model.model, optimizer, epoch, val_metrics,
                    output_dir / 'best_ema_model.pth', config
                )

            logger.info(f"New best model saved with val accuracy: {best_val_accuracy:.4f}")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= config['patience']:
            logger.info(f"Early stopping triggered after {epoch + 1} epochs")
            break

        # Save latest checkpoint
        if (epoch + 1) % config['checkpoint_interval'] == 0:
            save_checkpoint(
                model, optimizer, epoch, val_metrics,
                output_dir / f'checkpoint_epoch_{epoch + 1}.pth', config
            )

    logger.info("Training completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TRM model")
    parser.add_argument("--config", type=str, required=True,
                       help="Path to training config JSON file")
    parser.add_argument("--dataset", type=str, choices=['arc', 'sudoku', 'maze'],
                       help="Override dataset in config")
    parser.add_argument("--output_dir", type=str,
                       help="Override output directory")

    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)

    # Override config with command line args
    if args.dataset:
        config['dataset'] = args.dataset
    if args.output_dir:
        config['output_dir'] = args.output_dir

    # Train model
    train_model(config)