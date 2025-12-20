#!/usr/bin/env python3
"""
Samsung-style Multi-Task TRM Training.

Trains a single unified model on ARC, Sudoku, and Maze datasets simultaneously,
similar to Samsung's multi-task learning approach.
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
import psutil
import gc
import random
import numpy as np

from models.trm_multitask import MultiTaskTRM, create_multitask_trm, TASK_CONFIGS
from data.multitask_dataset import create_multitask_data_loader
from data.datasets import create_data_loader
from training.utils import EMA, save_checkpoint, load_checkpoint
from training.model_naming import generate_model_name, create_model_directory, save_model_metadata


class MultiTaskWarmupScheduler:
    """Multi-task aware warmup scheduler."""

    def __init__(self, optimizer, warmup_steps: int, base_lr: float):
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


def train_multitask_epoch(
    model: MultiTaskTRM,
    train_loader: DataLoader,
    optimizer,
    scheduler: MultiTaskWarmupScheduler,
    ema_model: Optional[EMA],
    device: torch.device,
    epoch: int,
    config: Dict[str, Any],
    logger: logging.Logger
) -> Dict[str, float]:
    """Train for one multi-task epoch."""
    model.train()
    total_loss = 0.0
    total_samples = 0
    task_losses = {}
    task_samples = {}

    for batch_idx, batch in enumerate(train_loader):
        # Handle mixed batches - group samples by task
        task_names = batch.get('dataset_name', ['arc'] * len(batch['input_tokens']))

        # Move batch to device
        input_tokens = batch['input_tokens'].to(device)
        target_tokens = batch['output_tokens'].to(device)

        batch_size = input_tokens.size(0)

        # Group samples by task
        task_groups = {}
        for sample_idx in range(batch_size):
            sample_task = task_names[sample_idx] if isinstance(task_names, list) and sample_idx < len(task_names) else 'arc'

            if sample_task not in task_groups:
                task_groups[sample_task] = {
                    'indices': [],
                    'inputs': [],
                    'targets': []
                }

            task_groups[sample_task]['indices'].append(sample_idx)
            task_groups[sample_task]['inputs'].append(input_tokens[sample_idx:sample_idx+1])
            task_groups[sample_task]['targets'].append(target_tokens[sample_idx:sample_idx+1])

        # Process each task group
        total_batch_loss = 0.0
        task_weight_sum = 0.0

        for task_name, group in task_groups.items():
            if not group['indices']:
                continue

            # Stack samples for this task
            task_inputs = torch.cat(group['inputs'], dim=0)
            task_targets_full = torch.cat(group['targets'], dim=0)

            # Task-specific target processing
            if task_name == 'arc':
                task_targets = task_targets_full[:, 0].long()  # First token prediction
            elif task_name == 'sudoku':
                task_targets = task_targets_full[:, 0].long()  # Digit prediction
            elif task_name == 'maze':
                task_targets = task_targets_full[:, 0].long()  # Path element prediction
            else:
                task_targets = task_targets_full[:, 0].long()

            # Get task-specific configuration
            task_config = config.get('task_configs', {}).get(task_name, TASK_CONFIGS.get(task_name, {}))
            K = task_config.get('recursion_depth', config.get('recursion_depth', 10))
            supervision_weight = task_config.get('supervision_weight', config.get('supervision_weight', 0.1))

            # Get task weight for loss scaling
            task_weight = config.get('multitask_weights', {}).get(task_name, 1.0)

            # Compute loss for this task group
            task_loss = model.get_loss(
                task_inputs,
                task_targets,
                task_name=task_name,
                K=K,
                supervision_weight=supervision_weight
            )

            # Apply task weighting and accumulate gradients
            weighted_loss = task_loss * task_weight

            # Backward pass for this task group
            weighted_loss.backward()

            total_batch_loss += weighted_loss.item()
            task_weight_sum += task_weight

            # Track task-specific statistics
            task_group_size = len(group['indices'])
            if task_name not in task_losses:
                task_losses[task_name] = 0.0
                task_samples[task_name] = 0

            task_losses[task_name] += task_loss.item() * task_group_size
            task_samples[task_name] += task_group_size

        # Average across task groups
        if task_weight_sum > 0:
            batch_loss = total_batch_loss / task_weight_sum
        else:
            batch_loss = total_batch_loss

        # Gradients already accumulated in task loops above

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])

        # Optimizer step
        optimizer.step()

        # Update EMA if available
        if ema_model is not None:
            ema_model.update(model)

        # Update scheduler
        scheduler.step()

        # Accumulate statistics
        total_loss += batch_loss * batch_size
        total_samples += batch_size

        # Logging
        if batch_idx % config['log_interval'] == 0:
            current_lr = scheduler.get_lr()
            # Show task distribution in batch
            task_counts = {}
            for task in task_names:
                task_counts[task] = task_counts.get(task, 0) + 1
            task_summary = ", ".join([f"{task}: {count}" for task, count in task_counts.items()])
            logger.info(
                f"Epoch {epoch + 1}, Batch {batch_idx}, Tasks: {task_summary}, "
                f"Loss: {batch_loss:.4f}, LR: {current_lr:.6f}"
            )

    # Compute final metrics
    avg_loss = total_loss / total_samples
    task_avg_losses = {
        task: task_losses[task] / task_samples[task]
        for task in task_losses.keys()
    }

    return {
        'train_loss': avg_loss,
        'task_losses': task_avg_losses,
        'total_samples': total_samples
    }


def validate_multitask(
    model: MultiTaskTRM,
    val_loaders: Dict[str, DataLoader],
    device: torch.device,
    config: Dict[str, Any]
) -> Dict[str, float]:
    """Validate on all tasks."""
    model.eval()
    results = {}

    with torch.no_grad():
        for task_name, val_loader in val_loaders.items():
            if val_loader is None:
                continue

            task_config = config.get('task_configs', {}).get(task_name, TASK_CONFIGS.get(task_name, {}))
            K = task_config.get('recursion_depth', config.get('recursion_depth', 10))

            total_loss = 0.0
            total_accuracy = 0.0
            num_batches = 0

            for batch in val_loader:
                input_tokens = batch['input_tokens'].to(device)
                target_tokens = batch['output_tokens'].to(device)

                # Task-specific target processing
                if task_name == 'arc':
                    targets = target_tokens[:, 0].long()
                elif task_name == 'sudoku':
                    targets = target_tokens[:, 0].long()
                elif task_name == 'maze':
                    targets = target_tokens[:, 0].long()
                else:
                    targets = target_tokens[:, 0].long()

                # Forward pass
                final_pred, _ = model(input_tokens, task_name, K=K)

                # Compute loss and accuracy
                if task_name == 'arc':
                    loss_fn = nn.CrossEntropyLoss()
                    loss = loss_fn(final_pred, targets)
                    _, predicted = torch.max(final_pred, dim=-1)
                    accuracy = (predicted == targets).float().mean()
                elif task_name == 'sudoku':
                    loss_fn = nn.CrossEntropyLoss()
                    loss = loss_fn(final_pred, targets)
                    _, predicted = torch.max(final_pred, dim=-1)
                    accuracy = (predicted == targets).float().mean()
                elif task_name == 'maze':
                    loss_fn = nn.CrossEntropyLoss()
                    loss = loss_fn(final_pred, targets)
                    _, predicted = torch.max(final_pred, dim=-1)
                    accuracy = (predicted == targets).float().mean()

                total_loss += loss.item()
                total_accuracy += accuracy.item()
                num_batches += 1

            avg_loss = total_loss / num_batches
            avg_accuracy = total_accuracy / num_batches

            results[f'{task_name}_loss'] = avg_loss
            results[f'{task_name}_accuracy'] = avg_accuracy

    return results


def get_memory_usage():
    """Get current memory usage in GB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 3)  # Convert to GB


def optimize_memory():
    """Force memory cleanup."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def train_multitask_model(config: Dict[str, Any]):
    """Train multi-task TRM model Samsung-style."""

    # Setup device and logging
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Set seeds for reproducibility
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])

    # Setup logging
    logger = logging.getLogger('multitask_trm')
    logger.setLevel(logging.INFO)

    # Generate model name and create directory
    model_name = generate_model_name(config, 'multitask')
    output_dir = create_model_directory(model_name, config['output_dir'])

    # Setup console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(message)s')  # Simpler format for console
    ch.setFormatter(console_formatter)
    logger.addHandler(ch)

    # Setup file handler
    fh = logging.FileHandler(output_dir / 'training.log')
    fh.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(file_formatter)
    logger.addHandler(fh)

    # Create multi-task data loader
    train_loader = create_multitask_data_loader(
        config['data_path'],
        split='train',
        datasets=config.get('multitask_datasets', ['arc', 'sudoku', 'maze']),
        dataset_weights=config.get('multitask_weights', {}),
        batch_size=config['batch_size'],
        shuffle=True
    )

    # Create validation loaders for each task
    val_loaders = {}
    for task_name in config.get('multitask_datasets', []):
        if config.get('multitask_weights', {}).get(task_name, 0) > 0:
            try:
                if task_name == 'maze':
                    # Skip maze validation for now due to architecture mismatch
                    val_loaders[task_name] = None
                else:
                    val_loader = create_data_loader(
                        task_name,
                        config['data_path'],
                        split='val',
                        batch_size=config['batch_size'],
                        shuffle=False,
                        max_samples=config.get('max_val_samples')
                    )
                    val_loaders[task_name] = val_loader
            except Exception as e:
                logger.warning(f"Could not create validation loader for {task_name}: {e}")
                val_loaders[task_name] = None

    # Create model
    model = create_multitask_trm(
        vocab_size=config.get('vocab_size', 12),
        embed_dim=config['embed_dim'],
        latent_dim=config['latent_dim'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        ff_dim=config['ff_dim'],
        num_tasks=len(config.get('multitask_datasets', [])),
        dropout=0.1
    ).to(device)

    logger.info(f"Multi-task TRM created with {sum(p.numel() for p in model.parameters()):,} parameters")
    logger.info(f"Training on tasks: {config.get('multitask_datasets', [])}")
    logger.info(f"Task weights: {config.get('multitask_weights', {})}")

    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
        betas=(0.9, 0.999)
    )

    # Create scheduler
    scheduler = MultiTaskWarmupScheduler(optimizer, config['warmup_steps'], config['learning_rate'])

    # Create EMA model
    ema_model = EMA(model, decay=config['ema_decay']) if config.get('use_ema', False) else None

    # Save model metadata
    save_model_metadata(output_dir, config)

    # Training loop
    best_avg_accuracy = 0.0
    patience_counter = 0

    for epoch in range(config['num_epochs']):
        logger.info(f"Starting epoch {epoch + 1}/{config['num_epochs']}")

        # Memory optimization and monitoring
        optimize_memory()
        memory_usage = get_memory_usage()
        logger.info(f"Memory usage before epoch {epoch + 1}: {memory_usage:.1f}GB")

        # Train epoch
        train_metrics = train_multitask_epoch(
            model, train_loader, optimizer, scheduler, ema_model,
            device, epoch, config, logger
        )

        # Log memory after epoch
        optimize_memory()
        memory_after = get_memory_usage()
        logger.info(f"Memory usage after epoch {epoch + 1}: {memory_after:.1f}GB")

        # Warn if memory is getting high
        if memory_after > 25:  # 25GB warning threshold
            logger.warning(f"High memory usage: {memory_after:.1f}GB - Consider reducing batch_size or max_seq_len")

        # Validate on all tasks
        val_metrics = validate_multitask(model, val_loaders, device, config)

        # Compute average accuracy across tasks
        task_accuracies = [v for k, v in val_metrics.items() if k.endswith('_accuracy')]
        avg_accuracy = sum(task_accuracies) / len(task_accuracies) if task_accuracies else 0.0

        # Log results
        logger.info(
            f"Epoch {epoch + 1} - Train Loss: {train_metrics['train_loss']:.4f}, "
            f"Avg Val Acc: {avg_accuracy:.4f}"
        )

        # Log individual task results
        for task_name in config.get('multitask_datasets', []):
            if f'{task_name}_accuracy' in val_metrics:
                logger.info(
                    f"  {task_name.upper()}: Loss={val_metrics[f'{task_name}_loss']:.4f}, "
                    f"Acc={val_metrics[f'{task_name}_accuracy']:.4f}"
                )

        # Save best model
        if avg_accuracy > best_avg_accuracy:
            best_avg_accuracy = avg_accuracy
            patience_counter = 0

            # Save regular model
            save_checkpoint(
                model, optimizer, epoch, val_metrics,
                output_dir / 'best_model.pth', config
            )

            # Save EMA model
            if ema_model is not None:
                save_checkpoint(
                    ema_model.model, optimizer, epoch, val_metrics,
                    output_dir / 'best_ema_model.pth', config
                )

            logger.info(f"New best model saved with avg accuracy: {best_avg_accuracy:.4f}")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= config['patience']:
            logger.info(f"Early stopping triggered after {epoch + 1} epochs")
            break

        # Save periodic checkpoints
        if (epoch + 1) % config['checkpoint_interval'] == 0:
            save_checkpoint(
                model, optimizer, epoch, val_metrics,
                output_dir / f'checkpoint_epoch_{epoch + 1}.pth', config
            )

    logger.info("Multi-task training completed!")
    logger.info(f"Best average validation accuracy: {best_avg_accuracy:.4f}")
    logger.info(f"Model saved in: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Multi-Task TRM Model")
    parser.add_argument("--config", type=str, required=True,
                       help="Path to training config JSON file")

    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)

    # Train model
    train_multitask_model(config)