#!/usr/bin/env python3
"""
Script to easily load and use trained TRM models.
"""

import sys
from pathlib import Path
import torch
import json

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from models.trm import TinyRecursiveModel
from training.model_naming import list_available_models, load_model_metadata


def load_trained_model(model_name_or_path, device='cpu'):
    """
    Load a trained TRM model by name or path.

    Args:
        model_name_or_path: Either a model name (from outputs/) or full path
        device: Device to load model on

    Returns:
        model: Loaded PyTorch model
        metadata: Model metadata dictionary
    """

    # Determine if it's a model name or path
    if Path(model_name_or_path).is_absolute() or (Path('outputs') / model_name_or_path).exists():
        if Path(model_name_or_path).is_absolute():
            model_dir = Path(model_name_or_path)
        else:
            model_dir = Path('outputs') / model_name_or_path
    else:
        # Try to find by name
        models = list_available_models()
        matching_models = [m for m in models if m['name'] == model_name_or_path]

        if not matching_models:
            print(f"Model '{model_name_or_path}' not found. Available models:")
            for model in models:
                print(f"  - {model['name']}")
            return None, None

        model_dir = Path(matching_models[0]['path'])

    # Load metadata
    metadata = load_model_metadata(model_dir.name)
    if not metadata:
        print(f"No metadata found for model in {model_dir}")
        return None, None

    # Load config
    config_path = model_dir / "config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Create model
    model = TinyRecursiveModel(
        vocab_size=config.get('vocab_size', 11),
        embed_dim=config['embed_dim'],
        latent_dim=config['latent_dim'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        ff_dim=config['ff_dim'],
        dropout=0.3  # Use consistent dropout
    )

    # Load checkpoint
    checkpoint_path = model_dir / "best_model.pth"
    if not checkpoint_path.exists():
        print(f"Model checkpoint not found: {checkpoint_path}")
        return None, None

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    print("Loaded model:")
    print(f"  Name: {metadata['model_name']}")
    print(f"  Created: {metadata['created_at']}")
    print(".1f")
    print(f"  Parameters: {metadata['config']['embed_dim']}d_{metadata['config']['latent_dim']}l")
    print(f"  Device: {device}")

    return model, metadata


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Load a trained TRM model")
    parser.add_argument("--model", type=str,
                       help="Model name or path")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device to load model on")
    parser.add_argument("--list", action="store_true",
                       help="List all available models")

    args = parser.parse_args()

    if args.list:
        models = list_available_models()
        print("Available trained models:")
        for model in models:
            stats = model.get('training_stats', {})
            acc = stats.get('final_val_accuracy', 'unknown')
            print(".1f")
        return

    if not args.model:
        print("Please specify a model name with --model")
        return

    model, metadata = load_trained_model(args.model, args.device)

    if model is not None:
        print("\nModel ready for inference!")
        print(f"Use model.forward(input, K={metadata['config']['recursion_depth']}) for predictions")
    else:
        print("Failed to load model")


if __name__ == "__main__":
    main()