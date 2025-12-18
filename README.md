# Tiny Recursive Model (TRM) Implementation

This repository contains a complete PyTorch implementation of Samsung's Tiny Recursive Model (TRM) for reasoning tasks, as described in "Less is More: Recursive Reasoning with Tiny Networks".

TRM is a lightweight recursive architecture that excels at abstract reasoning tasks including ARC-AGI puzzles, Sudoku solving, and maze pathfinding.

## Features

- **Complete TRM Architecture**: 2-layer transformer with recursive latent state updates
- **Multiple Datasets**: ARC-AGI, Sudoku-Extreme, and Maze-Hard with data augmentation
- **Advanced Training**: AdamW optimizer, warmup scheduling, EMA, deep supervision
- **Comprehensive Evaluation**: Accuracy metrics and benchmarking tools
- **Easy Configuration**: JSON-based config system for different datasets

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd TRM
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Download Datasets

Download and prepare the datasets:

```bash
# Download all datasets
python -m data.download_datasets

# Or download specific datasets
python -c "from data.download_datasets import ARCDownloader; ARCDownloader().download()"
```

### 2. Train a Model

Train on ARC-AGI dataset:

```bash
# Generate config
python -m training.config --dataset arc --output config_arc.json

# Train model
python -m training.train --config config_arc.json
```

### 3. Evaluate Model

Evaluate trained model:

```bash
# Evaluate on all datasets
python -m evaluation.evaluate --checkpoint outputs/best_model.pth --output results.json

# Benchmark different recursion depths
python -m evaluation.evaluate --checkpoint outputs/best_model.pth --benchmark_depths --dataset arc
```

## Project Structure

```
TRM/
├── models/                 # Model architectures
│   ├── __init__.py
│   └── trm.py             # TinyRecursiveModel implementation
├── data/                   # Dataset handling
│   ├── __init__.py
│   ├── datasets.py        # Dataset loaders with augmentation
│   └── download_datasets.py # Dataset download scripts
├── training/              # Training infrastructure
│   ├── __init__.py
│   ├── train.py           # Main training script
│   ├── config.py          # Training configurations
│   └── utils.py           # Training utilities (EMA, checkpointing)
├── evaluation/            # Evaluation tools
│   ├── __init__.py
│   └── evaluate.py        # Evaluation and benchmarking
├── requirements.txt       # Dependencies
└── README.md             # This file
```

## Model Architecture

The TRM consists of:

- **2-layer Transformer**: Self-attention mechanism with feed-forward networks
- **Recursive Reasoning**: K-step iterative updates of latent state z from (x, y, z)
- **Tokenization**: Flattened 2D grids with learned positional embeddings
- **Deep Supervision**: Loss applied at intermediate recursion steps

### Key Hyperparameters

- `recursion_depth` (K): 10-16 steps for different tasks
- `latent_dim`: 64-dimensional latent state
- `embed_dim`: 128-dimensional token embeddings
- Learning rate: 1e-4 with AdamW and weight decay 1e-2

## Datasets

### ARC-AGI
- **Source**: Official ARC challenge dataset
- **Task**: Abstract reasoning puzzles
- **Augmentation**: Rotations, reflections, grid transformations
- **Config**: Smaller batches (16), longer training

### Sudoku-Extreme
- **Source**: Generated synthetic puzzles
- **Task**: 9×9 Sudoku solving
- **Augmentation**: Row/column shuffling within boxes
- **Config**: Standard batch size (32), moderate recursion

### Maze-Hard
- **Source**: Generated pathfinding mazes
- **Task**: 30×30 maze navigation
- **Augmentation**: Geometric transformations
- **Config**: Smaller batches (16), deeper recursion (16 steps)

## Training

### Single Dataset Training

```python
from training.config import get_arc_config
from training.train import train_model

config = get_arc_config()
config['output_dir'] = 'outputs/arc_experiment'
train_model(config)
```

### Multi-Task Training

```python
from training.config import get_multi_task_config
# Modify config for multi-task learning
```

### Training Features

- **AdamW Optimizer**: With weight decay and beta parameters
- **Warmup Scheduling**: Linear warmup over first 200-500 steps
- **EMA**: Exponential moving average for stable training
- **Deep Supervision**: Auxiliary losses at recursion steps
- **Early Stopping**: Patience-based stopping with best model saving

## Evaluation

### Metrics

- **Token Accuracy**: Per-token prediction accuracy
- **Sequence Accuracy**: Full sequence prediction accuracy
- **Loss**: Cross-entropy loss across recursion steps

### Benchmarking

```python
from evaluation.evaluate import evaluate_checkpoint

results = evaluate_checkpoint(
    checkpoint_path='outputs/best_model.pth',
    data_path='data',
    output_path='evaluation_results.json'
)
```

## Configuration

Training configurations are defined in `training/config.py`:

```python
# ARC-AGI config
config = {
    "dataset": "arc",
    "batch_size": 16,
    "recursion_depth": 10,
    "learning_rate": 1e-4,
    "num_epochs": 50,
    # ... other parameters
}
```

## Advanced Usage

### Custom Dataset

Create a new dataset loader by extending the base classes in `data/datasets.py`:

```python
class CustomDataset(Dataset):
    def __getitem__(self, idx):
        # Return input_tokens, input_mask, output_tokens, output_mask
        pass
```

### Model Modifications

Extend the TRM architecture in `models/trm.py`:

```python
class CustomTRM(TinyRecursiveModel):
    def forward(self, x, K=10):
        # Custom forward logic
        pass
```

## Citation

If you use this implementation, please cite the original TRM paper:

```
@article{TRM2024,
  title={Less is More: Recursive Reasoning with Tiny Networks},
  author={Samsung AI Research},
  year={2024}
}
```

## License

This implementation is provided for research purposes. Please check the original dataset licenses for commercial use.