#!/usr/bin/env python3
"""
Evaluation script for Multi-Task TRM models on ARC, Sudoku, and Maze datasets.

This script evaluates trained multi-task TRM models that can handle multiple reasoning tasks
simultaneously, similar to Samsung's multi-task learning approach.
"""

import sys
from pathlib import Path
import json
import numpy as np
import torch
import argparse
import urllib.request
import zipfile
import os

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.trm_multitask import MultiTaskTRM, create_multitask_trm, TASK_CONFIGS
from data.datasets import create_data_loader
from training.utils import load_checkpoint


class MultiTaskEvaluator:
    """Evaluate Multi-Task TRM model on multiple reasoning datasets."""

    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.model.eval()

    def evaluate_task(self, task_name: str, data_path="data", split="val", max_samples=1000):
        """Evaluate on a specific task."""
        print(f"\nEvaluating on {task_name.upper()}-{split.upper()} set...")

        try:
            val_loader = create_data_loader(
                dataset_name=task_name,
                data_path=data_path,
                split=split,
                batch_size=8,
                shuffle=False,
                num_workers=0,
                max_samples=max_samples
            )

            print(f"   Dataset: {len(val_loader.dataset)} tasks")

            # Get task-specific configuration
            task_config = TASK_CONFIGS.get(task_name, {})
            K = task_config.get('recursion_depth', 10)

            total_loss = 0.0
            total_accuracy = 0.0
            num_batches = 0

            with torch.no_grad():
                for batch_idx, batch in enumerate(val_loader):
                    if batch_idx % 10 == 0:
                        print(f"   Processed {batch_idx}/{len(val_loader)} batches...")

                    # Move batch to device
                    input_tokens = batch['input_tokens'].to(self.device)
                    target_tokens = batch['output_tokens'].to(self.device)

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
                    final_pred, _ = self.model(input_tokens, task_name, K=K)

                    # Compute loss and accuracy
                    loss_fn = torch.nn.CrossEntropyLoss()
                    loss = loss_fn(final_pred, targets)
                    _, predicted = torch.max(final_pred, dim=-1)
                    accuracy = (predicted == targets).float().mean()

                    total_loss += loss.item()
                    total_accuracy += accuracy.item()
                    num_batches += 1

            avg_loss = total_loss / num_batches
            avg_accuracy = total_accuracy / num_batches

            print("   Results:")
            print(f"   Loss: {avg_loss:.4f}")
            print(f"   Accuracy: {avg_accuracy:.4f}")
            return {
                "dataset": f"{task_name.upper()}-{split.upper()}",
                "tasks": len(val_loader.dataset),
                "accuracy": avg_accuracy,
                "loss": avg_loss
            }

        except Exception as e:
            print(f"   Error evaluating {task_name}: {e}")
            return None

    def evaluate_all(self, data_path="data", tasks=['arc', 'sudoku'], max_samples=1000):
        """Evaluate on all available tasks."""
        print("Starting multi-task evaluation...")

        results = []

        for task_name in tasks:
            if task_name == 'maze':
                print(f"Skipping maze evaluation (architecture not suitable)")
                continue

            result = self.evaluate_task(task_name, data_path, "val", max_samples)
            if result:
                results.append(result)

        return results


def load_multitask_model(model_path, device='cpu'):
    """Load a trained multi-task TRM model."""

    checkpoint_path = Path(model_path)
    if not checkpoint_path.exists():
        print(f"Model checkpoint not found: {checkpoint_path}")
        return None, None

    # Load checkpoint to get config
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if 'config' not in checkpoint:
        print("No config found in checkpoint. Using default config.")
        # Default config for multitask 64d 32l
        config = {
            'vocab_size': 12,
            'embed_dim': 64,
            'latent_dim': 32,
            'num_layers': 2,
            'num_heads': 4,
            'ff_dim': 256,
            'num_tasks': 3,
            'dropout': 0.1
        }
    else:
        config = checkpoint['config']

    # Create model
    model = create_multitask_trm(
        vocab_size=config.get('vocab_size', 12),
        embed_dim=config.get('embed_dim', 64),
        latent_dim=config.get('latent_dim', 32),
        num_layers=config.get('num_layers', 2),
        num_heads=config.get('num_heads', 4),
        ff_dim=config.get('ff_dim', 256),
        num_tasks=config.get('num_tasks', 3),
        dropout=config.get('dropout', 0.1)
    )

    # Load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    print("Loaded multi-task model:")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Embed dim: {config.get('embed_dim', 64)}")
    print(f"  Latent dim: {config.get('latent_dim', 32)}")
    print(f"  Device: {device}")

    return model, config


def download_arc_agi_2(output_dir="data/arc2"):
    """Download ARC-AGI-2 dataset using Hugging Face datasets library."""
    print("Downloading ARC-AGI-2 dataset using Hugging Face...")

    try:
        from datasets import load_dataset
    except ImportError:
        print("   Error: datasets library not installed.")
        print("   Please install with: pip install datasets")
        return

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    filenames = ["training.jsonl", "evaluation.jsonl"]

    try:
        print("   Loading ARC-AGI-2 dataset from Hugging Face...")
        dataset = load_dataset("eturok/ARC-AGI-2")

        # Save training set
        train_filepath = output_path / "training.jsonl"
        if not train_filepath.exists():
            print("   Saving training set...")
            with open(train_filepath, 'w') as f:
                for example in dataset['train']:
                    json.dump(example, f)
                    f.write('\n')
            print(f"   Saved training set with {len(dataset['train'])} examples")

        # Save evaluation set
        eval_filepath = output_path / "evaluation.jsonl"
        if not eval_filepath.exists():
            print("   Saving evaluation set...")
            with open(eval_filepath, 'w') as f:
                for example in dataset['eval']:
                    json.dump(example, f)
                    f.write('\n')
            print(f"   Saved evaluation set with {len(dataset['eval'])} examples")

        print("ARC-AGI-2 download complete!")

    except Exception as e:
        print(f"   Error downloading ARC-AGI-2: {e}")
        print("   You can also try manual download from:")
        print("   https://huggingface.co/datasets/eturok/ARC-AGI-2")


def main():
    parser = argparse.ArgumentParser(description="Multi-Task TRM Evaluation")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Path to multi-task model checkpoint")
    parser.add_argument("--data_path", type=str, default="data",
                       help="Path to data directory")
    parser.add_argument("--output", type=str, default="multitask_results.json",
                       help="Output JSON file")
    parser.add_argument("--tasks", nargs="+", choices=["arc", "sudoku", "maze"],
                       default=["arc", "sudoku"],
                       help="Tasks to evaluate on")
    parser.add_argument("--max_samples", type=int, default=1000,
                       help="Maximum samples per task for evaluation")
    parser.add_argument("--download_arc2", action="store_true",
                       help="Download ARC-AGI-2 dataset")
    parser.add_argument("--arc2_dir", type=str, default="data/arc2",
                       help="Directory to save ARC-AGI-2 dataset")

    args = parser.parse_args()

    # Download ARC-AGI-2 if requested
    if args.download_arc2:
        download_arc_agi_2(args.arc2_dir)

    # If only downloading, exit
    if args.download_arc2 and not args.checkpoint:
        print("ARC-AGI-2 download complete!")
        return

    # Check if checkpoint is provided
    if not args.checkpoint:
        print("Error: --checkpoint is required for evaluation")
        return

    # Load model
    print("Loading multi-task model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, config = load_multitask_model(args.checkpoint, device)

    if model is None:
        print("Failed to load model")
        return

    # Create evaluator
    evaluator = MultiTaskEvaluator(model, device)

    # Evaluate on specified tasks
    all_results = evaluator.evaluate_all(args.data_path, args.tasks, args.max_samples)

    # Save comprehensive results
    final_results = {
        "model": "Multi-Task TRM",
        "checkpoint": str(args.checkpoint),
        "model_config": config,
        "tasks_evaluated": args.tasks,
        "results": all_results,
        "summary": {
            "datasets_evaluated": len(all_results),
            "best_accuracy": max([r["accuracy"] for r in all_results]) if all_results else 0,
            "average_accuracy": sum([r["accuracy"] for r in all_results]) / len(all_results) if all_results else 0
        }
    }

    with open(args.output, 'w') as f:
        json.dump(final_results, f, indent=2)

    print("\nFinal Summary:")
    for result in all_results:
        print(".4f")
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()