"""
Training configurations for TRM models.

Contains default hyperparameters and dataset-specific configs.
"""

from typing import Dict, Any


def get_default_config() -> Dict[str, Any]:
    """Get default training configuration."""
    return {
        # Model architecture
        "embed_dim": 128,
        "num_layers": 2,
        "num_heads": 8,
        "ff_dim": 512,
        "latent_dim": 64,
        "max_seq_len": 1024,

        # Training hyperparameters
        "learning_rate": 1e-4,
        "weight_decay": 1e-2,
        "batch_size": 32,
        "num_epochs": 100,
        "warmup_steps": 200,
        "gradient_accumulation_steps": 1,
        "max_grad_norm": 1.0,

        # TRM-specific parameters
        "recursion_depth": 12,  # K = 10-16 steps
        "inner_steps": 6,       # n = 6 per recursive block
        "supervision_weight": 0.1,  # Deep supervision weight

        # EMA parameters
        "use_ema": True,
        "ema_decay": 0.999,

        # Training settings
        "seed": 42,
        "log_interval": 10,
        "checkpoint_interval": 10,
        "patience": 10,  # Early stopping patience

        # Data settings
        "max_train_samples": None,
        "max_val_samples": None,
        "data_path": "data",

        # Output settings
        "output_dir": "outputs",
        "experiment_name": "trm_training"
    }


def get_arc_config() -> Dict[str, Any]:
    """Configuration optimized for ARC-AGI dataset."""
    config = get_default_config()
    config.update({
        "dataset": "arc",
        "batch_size": 16,  # Smaller batch for ARC generalization
        "recursion_depth": 10,
        "max_seq_len": 1024,  # ARC grids can be various sizes
        "learning_rate": 1e-4,
        "warmup_steps": 100,
        "num_epochs": 50,
        "experiment_name": "trm_arc"
    })
    return config


def get_sudoku_config() -> Dict[str, Any]:
    """Configuration optimized for Sudoku-Extreme dataset."""
    config = get_default_config()
    config.update({
        "dataset": "sudoku",
        "batch_size": 32,
        "recursion_depth": 12,
        "max_seq_len": 81,  # 9x9 grid flattened
        "learning_rate": 1e-4,
        "warmup_steps": 200,
        "num_epochs": 30,
        "experiment_name": "trm_sudoku"
    })
    return config


def get_maze_config() -> Dict[str, Any]:
    """Configuration optimized for Maze-Hard dataset."""
    config = get_default_config()
    config.update({
        "dataset": "maze",
        "batch_size": 16,
        "recursion_depth": 16,  # More steps for pathfinding
        "max_seq_len": 900,  # 30x30 grid flattened
        "learning_rate": 1e-4,
        "warmup_steps": 300,
        "num_epochs": 40,
        "experiment_name": "trm_maze"
    })
    return config


def get_multi_task_config() -> Dict[str, Any]:
    """Configuration for multi-task training across datasets."""
    config = get_default_config()
    config.update({
        "dataset": "arc",  # Primary dataset, will be overridden in training loop
        "batch_size": 16,  # Conservative batch size
        "recursion_depth": 12,
        "max_seq_len": 1024,
        "learning_rate": 1e-4,
        "warmup_steps": 500,  # Longer warmup for multi-task
        "num_epochs": 100,
        "experiment_name": "trm_multitask"
    })
    return config


def get_config_for_dataset(dataset_name: str) -> Dict[str, Any]:
    """Get configuration for specific dataset."""
    configs = {
        "arc": get_arc_config,
        "sudoku": get_sudoku_config,
        "maze": get_maze_config,
        "multitask": get_multi_task_config
    }

    if dataset_name not in configs:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(configs.keys())}")

    return configs[dataset_name]()


def save_config(config: Dict[str, Any], filepath: str) -> None:
    """Save configuration to JSON file."""
    import json
    from pathlib import Path

    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"Configuration saved to {filepath}")


def load_config(filepath: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    import json

    with open(filepath, 'r') as f:
        config = json.load(f)

    return config


# Pre-defined config files for easy access
ARC_CONFIG = get_arc_config()
SUDOKU_CONFIG = get_sudoku_config()
MAZE_CONFIG = get_maze_config()
MULTITASK_CONFIG = get_multi_task_config()


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Generate TRM training configs")
    parser.add_argument("--dataset", type=str, choices=['arc', 'sudoku', 'maze', 'multitask'],
                       help="Dataset to generate config for")
    parser.add_argument("--output", type=str, help="Output config file path")

    args = parser.parse_args()

    if args.dataset:
        config = get_config_for_dataset(args.dataset)
        if args.output:
            save_config(config, args.output)
        else:
            print(json.dumps(config, indent=2))
    else:
        # Print all configs
        print("ARC Config:")
        print(json.dumps(get_arc_config(), indent=2))
        print("\nSudoku Config:")
        print(json.dumps(get_sudoku_config(), indent=2))
        print("\nMaze Config:")
        print(json.dumps(get_maze_config(), indent=2))