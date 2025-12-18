"""
Model naming conventions for TRM experiments.
"""

from datetime import datetime
from pathlib import Path
import json


def generate_model_name(config: dict, experiment_type: str = "single") -> str:
    """
    Generate a systematic model name based on configuration.

    Naming convention: TRM_{experiment}_{timestamp}_{key_params}
    """

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    if experiment_type == "multitask":
        # Multi-task model naming
        datasets = "_".join(config.get("multitask_datasets", ["unknown"]))
        embed_dim = config.get("embed_dim", "unknown")
        latent_dim = config.get("latent_dim", "unknown")

        model_name = f"TRM_multitask_{datasets}_{embed_dim}d_{latent_dim}l_{timestamp}"

    elif experiment_type == "single":
        # Single-task model naming
        dataset = config.get("dataset", "unknown")
        embed_dim = config.get("embed_dim", "unknown")
        latent_dim = config.get("latent_dim", "unknown")

        model_name = f"TRM_{dataset}_{embed_dim}d_{latent_dim}l_{timestamp}"

    else:
        # Fallback naming
        model_name = f"TRM_{experiment_type}_{timestamp}"

    return model_name


def create_model_directory(model_name: str, output_dir: str = "outputs") -> Path:
    """Create and return model directory path."""
    model_dir = Path(output_dir) / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


def save_model_metadata(model_dir: Path, config: dict, training_stats: dict = None):
    """Save model metadata and configuration."""
    metadata = {
        "model_name": model_dir.name,
        "created_at": datetime.now().isoformat(),
        "config": config,
        "training_stats": training_stats or {},
        "model_paths": {
            "checkpoint": str(model_dir / "model.pth"),
            "config": str(model_dir / "config.json"),
            "training_log": str(model_dir / "training.log")
        }
    }

    with open(model_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    # Also save config separately
    with open(model_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)


def load_model_metadata(model_name: str, output_dir: str = "outputs") -> dict:
    """Load model metadata."""
    metadata_path = Path(output_dir) / model_name / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            return json.load(f)
    return {}


def list_available_models(output_dir: str = "outputs") -> list:
    """List all available trained models."""
    output_path = Path(output_dir)
    if not output_path.exists():
        return []

    models = []
    for item in output_path.iterdir():
        if item.is_dir() and (item / "model.pth").exists():
            metadata = load_model_metadata(item.name, output_dir)
            models.append({
                "name": item.name,
                "path": str(item),
                "created_at": metadata.get("created_at", "unknown"),
                "config": metadata.get("config", {}),
                "training_stats": metadata.get("training_stats", {})
            })

    return sorted(models, key=lambda x: x.get("created_at", ""), reverse=True)


if __name__ == "__main__":
    # Test model naming
    config_multitask = {
        "embed_dim": 64,
        "latent_dim": 32,
        "multitask_datasets": ["arc", "sudoku", "maze"],
        "dataset": "multitask"
    }

    model_name = generate_model_name(config_multitask, "multitask")
    print(f"Generated model name: {model_name}")

    # Test directory creation
    model_dir = create_model_directory(model_name)
    print(f"Model directory: {model_dir}")

    # Save metadata
    save_model_metadata(model_dir, config_multitask, {"epochs": 50, "final_accuracy": 0.75})
    print(f"Metadata saved to {model_dir}/metadata.json")

    # List models
    models = list_available_models()
    print(f"Available models: {len(models)}")
    for model in models[:3]:  # Show first 3
        print(f"  - {model['name']} ({model.get('created_at', 'unknown')})")