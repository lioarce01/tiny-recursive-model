#!/usr/bin/env python3
"""
Comprehensive evaluation script for TRM model on ARC and Sudoku datasets.
Maze evaluation removed - TRM architecture not suitable for maze navigation tasks.
Also includes functionality to download ARC-AGI-2 dataset.
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
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from models.trm import TinyRecursiveModel
from data.datasets import create_data_loader
from training.utils import load_checkpoint


class ComprehensiveEvaluator:
    """Evaluate TRM model on multiple reasoning datasets."""

    def __init__(self, model, device='cpu', recursion_depth=5):
        self.model = model.to(device)
        self.device = device
        self.recursion_depth = recursion_depth
        self.model.eval()

    def evaluate_arc(self, data_path="data", split="val"):
        """Evaluate on ARC-AGI dataset."""
        print(f"\nEvaluating on ARC-{split.upper()} set...")

        try:
            val_loader = create_data_loader(
                dataset_name="arc",
                data_path=data_path,
                split=split,
                batch_size=8,
                shuffle=False,
                num_workers=0
            )

            print(f"   Dataset: {len(val_loader.dataset)} tasks")

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
                    targets = target_tokens[:, 0].long()

                    # Forward pass
                    batch_loss = self.model.get_loss(input_tokens, targets, K=self.recursion_depth)
                    predictions = self.model(input_tokens, K=self.recursion_depth)

                    # Compute accuracy
                    final_pred = predictions[0]
                    if final_pred.dim() == 3:
                        _, predicted = torch.max(final_pred[:, 0, :], dim=-1)
                    else:
                        _, predicted = torch.max(final_pred, dim=-1)

                    batch_accuracy = (predicted == targets).float().mean()

                    total_loss += batch_loss.item()
                    total_accuracy += batch_accuracy.item()
                    num_batches += 1

            avg_loss = total_loss / num_batches
            avg_accuracy = total_accuracy / num_batches

            print("   Results:")
            print(f"   Loss: {avg_loss:.4f}")
            print(f"   Accuracy: {avg_accuracy:.4f}")
            return {
                "dataset": f"ARC-{split.upper()}",
                "tasks": len(val_loader.dataset),
                "accuracy": avg_accuracy,
                "loss": avg_loss
            }

        except Exception as e:
            print(f"   Error evaluating ARC: {e}")
            return None

    def evaluate_maze(self, data_path="data", max_samples=1000):
        """Evaluate on Maze-Hard dataset."""
        print(f"\nEvaluating on Maze dataset...")

        try:
            val_loader = create_data_loader(
                dataset_name="maze",
                data_path=data_path,
                split="val",
                batch_size=8,
                shuffle=False,
                num_workers=0,
                max_samples=max_samples
            )

            print(f"   Dataset: {len(val_loader.dataset)} mazes")

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
                    targets = target_tokens[:, 0].long()

                    # Forward pass
                    batch_loss = self.model.get_loss(input_tokens, targets, K=self.recursion_depth)
                    predictions = self.model(input_tokens, K=self.recursion_depth)

                    # Compute accuracy
                    final_pred = predictions[0]
                    if final_pred.dim() == 3:
                        _, predicted = torch.max(final_pred[:, 0, :], dim=-1)
                    else:
                        _, predicted = torch.max(final_pred, dim=-1)

                    batch_accuracy = (predicted == targets).float().mean()

                    total_loss += batch_loss.item()
                    total_accuracy += batch_accuracy.item()
                    num_batches += 1

            avg_loss = total_loss / num_batches
            avg_accuracy = total_accuracy / num_batches

            print("   Results:")
            print(f"   Loss: {avg_loss:.4f}")
            print(f"   Accuracy: {avg_accuracy:.4f}")
            return {
                "dataset": "Maze-Hard",
                "tasks": len(val_loader.dataset),
                "accuracy": avg_accuracy,
                "loss": avg_loss
            }

        except Exception as e:
            print(f"   Error evaluating Maze: {e}")
            return None

    def evaluate_sudoku(self, data_path="data", max_samples=1000):
        """Evaluate on Sudoku-Extreme dataset."""
        print(f"\nEvaluating on Sudoku dataset...")

        try:
            val_loader = create_data_loader(
                dataset_name="sudoku",
                data_path=data_path,
                split="val",
                batch_size=8,
                shuffle=False,
                num_workers=0,
                max_samples=max_samples
            )

            print(f"   Dataset: {len(val_loader.dataset)} puzzles")

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
                    targets = target_tokens[:, 0].long()

                    # Forward pass
                    batch_loss = self.model.get_loss(input_tokens, targets, K=self.recursion_depth)
                    predictions = self.model(input_tokens, K=self.recursion_depth)

                    # Compute accuracy
                    final_pred = predictions[0]
                    if final_pred.dim() == 3:
                        _, predicted = torch.max(final_pred[:, 0, :], dim=-1)
                    else:
                        _, predicted = torch.max(final_pred, dim=-1)

                    batch_accuracy = (predicted == targets).float().mean()

                    total_loss += batch_loss.item()
                    total_accuracy += batch_accuracy.item()
                    num_batches += 1

            avg_loss = total_loss / num_batches
            avg_accuracy = total_accuracy / num_batches

            print("   Results:")
            print(f"   Loss: {avg_loss:.4f}")
            print(f"   Accuracy: {avg_accuracy:.4f}")
            return {
                "dataset": "Sudoku-Extreme",
                "tasks": len(val_loader.dataset),
                "accuracy": avg_accuracy,
                "loss": avg_loss
            }

        except Exception as e:
            print(f"   Error evaluating Sudoku: {e}")
            return None

    def evaluate_all(self, data_path="data"):
        """Evaluate on all available datasets."""
        print("Starting comprehensive evaluation...")

        results = []

        # Evaluate ARC
        arc_result = self.evaluate_arc(data_path, "val")
        if arc_result:
            results.append(arc_result)

        # Evaluate Maze
        maze_result = self.evaluate_maze(data_path)
        if maze_result:
            results.append(maze_result)

        # Evaluate Sudoku
        sudoku_result = self.evaluate_sudoku(data_path)
        if sudoku_result:
            results.append(sudoku_result)

        return results


def download_arc_agi_2(output_dir="data/arc2"):
    """Download ARC-AGI-2 dataset."""
    print("Downloading ARC-AGI-2 dataset...")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ARC-AGI-2 download URLs from GitHub
    urls = [
        "https://raw.githubusercontent.com/arcprize/ARC-AGI-2/main/data/training/train.jsonl",
        "https://raw.githubusercontent.com/arcprize/ARC-AGI-2/main/data/evaluation/eval.jsonl"
    ]

    filenames = ["training.jsonl", "evaluation.jsonl"]

    for url, filename in zip(urls, filenames):
        filepath = output_path / filename

        if filepath.exists():
            print(f"   {filename} already exists, skipping...")
            continue

        try:
            print(f"   Downloading {filename}...")
            urllib.request.urlretrieve(url, filepath)
            print(f"   Downloaded {filename}")
        except Exception as e:
            print(f"   Failed to download {filename}: {e}")
            # Try alternative Hugging Face URLs
            try:
                hf_urls = [
                    "https://huggingface.co/datasets/eturok/ARC-AGI-2/resolve/main/train.jsonl",
                    "https://huggingface.co/datasets/eturok/ARC-AGI-2/resolve/main/eval.jsonl"
                ]
                alt_url = hf_urls[0] if "training" in filename else hf_urls[1]
                print(f"   Trying alternative URL for {filename}...")
                urllib.request.urlretrieve(alt_url, filepath)
                print(f"   Downloaded {filename} from alternative source")
            except Exception as e2:
                print(f"   Failed to download {filename} from all sources: {e2}")

    print("ARC-AGI-2 download complete!")


def main():
    parser = argparse.ArgumentParser(description="Comprehensive TRM Evaluation")
    parser.add_argument("--checkpoint", type=str, default="outputs/best_model.pth",
                       help="Path to model checkpoint")
    parser.add_argument("--data_path", type=str, default="data",
                       help="Path to data directory")
    parser.add_argument("--output", type=str, default="comprehensive_results.json",
                       help="Output JSON file")
    parser.add_argument("--datasets", nargs="+", choices=["arc", "sudoku", "all"],
                       default=["all"], help="Datasets to evaluate on (maze not supported)")
    parser.add_argument("--download_arc2", action="store_true",
                       help="Download ARC-AGI-2 dataset")
    parser.add_argument("--arc2_dir", type=str, default="data/arc2",
                       help="Directory to save ARC-AGI-2 dataset")

    args = parser.parse_args()

    # Download ARC-AGI-2 if requested
    if args.download_arc2:
        download_arc_agi_2(args.arc2_dir)

    # Load model
    print("Loading model and checkpoint...")
    vocab_size = 11  # ARC colors 0-10
    model = TinyRecursiveModel(
        vocab_size=vocab_size,
        embed_dim=64,
        latent_dim=32,
        num_layers=2,
        num_heads=4,
        ff_dim=256,
        dropout=0.3
    )

    checkpoint_path = Path(args.checkpoint)
    epoch, metrics, config = load_checkpoint(checkpoint_path, model)
    print(f"Loaded checkpoint from epoch {epoch}")

    # Create evaluator
    device = torch.device('cpu')
    evaluator = ComprehensiveEvaluator(model, device, recursion_depth=5)

    # Evaluate on requested datasets
    all_results = []

    if "all" in args.datasets or "arc" in args.datasets:
        arc_result = evaluator.evaluate_arc(args.data_path, "val")
        if arc_result:
            all_results.append(arc_result)

    # Maze evaluation removed - TRM architecture not suitable for maze navigation tasks

    if "all" in args.datasets or "sudoku" in args.datasets:
        sudoku_result = evaluator.evaluate_sudoku(args.data_path)
        if sudoku_result:
            all_results.append(sudoku_result)

    # Save comprehensive results
    final_results = {
        "model": "TRM (Tiny Recursive Model)",
        "checkpoint": str(checkpoint_path),
        "epoch": epoch,
        "recursion_depth": 5,
        "model_params": "257K",
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