"""
Samsung-style Multi-Task TRM Architecture.

This implements a unified TRM model that can handle multiple reasoning tasks
(ARC, Sudoku, Maze) simultaneously, similar to Samsung's approach.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple
import math


class TaskEmbedding(nn.Module):
    """Task-specific embedding to condition the model on different tasks."""

    def __init__(self, num_tasks: int, embed_dim: int):
        super().__init__()
        self.task_embeddings = nn.Embedding(num_tasks, embed_dim)

    def forward(self, task_ids: torch.Tensor) -> torch.Tensor:
        """Get task embeddings for conditioning."""
        return self.task_embeddings(task_ids)


class MultiTaskTRM(nn.Module):
    """
    Multi-Task Tiny Recursive Model - Samsung style architecture.

    Features:
    - Shared transformer backbone
    - Task-specific conditioning
    - Unified vocabulary across tasks
    - Recursive reasoning with task awareness
    """

    def __init__(
        self,
        vocab_size: int = 12,  # Unified vocab: 0-9 colors + 10(empty) + 11(special)
        max_seq_len: int = 1024,
        embed_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 8,
        ff_dim: int = 512,
        latent_dim: int = 64,
        num_tasks: int = 3,  # ARC, Sudoku, Maze
        dropout: float = 0.1
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.num_tasks = num_tasks

        # Task embeddings for conditioning
        self.task_embedding = TaskEmbedding(num_tasks, embed_dim)

        # Token and positional embeddings (shared across tasks)
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)

        # Shared transformer backbone
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=ff_dim,
                dropout=dropout,
                batch_first=True
            ) for _ in range(num_layers)
        ])

        # Task-specific latent state initialization
        self.latent_init = nn.Parameter(torch.randn(num_tasks, latent_dim))

        # Recursive reasoning networks (shared but task-conditioned)
        self.update_z_net = nn.Sequential(
            nn.Linear(embed_dim + embed_dim + latent_dim + embed_dim, ff_dim),  # +embed_dim for task conditioning
            nn.ReLU(),
            nn.Linear(ff_dim, latent_dim)
        )

        self.update_y_net = nn.Sequential(
            nn.Linear(embed_dim + latent_dim + embed_dim, ff_dim),  # +embed_dim for task conditioning
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )

        # Task-specific prediction heads
        self.task_heads = nn.ModuleDict({
            'arc': nn.Linear(embed_dim, vocab_size),      # ARC: predict next token
            'sudoku': nn.Linear(embed_dim, 10),           # Sudoku: predict digit 1-9 (0=empty)
            'maze': nn.Linear(embed_dim, 5)               # Maze: predict path elements
        })

    def get_task_id(self, task_name: str) -> int:
        """Convert task name to ID."""
        task_map = {'arc': 0, 'sudoku': 1, 'maze': 2}
        return task_map.get(task_name, 0)

    def _get_embeddings(self, x: torch.Tensor, task_id: int) -> torch.Tensor:
        """Get token, positional, and task embeddings."""
        seq_len = x.size(1)

        # Token embeddings
        token_embeds = self.token_embedding(x)

        # Positional embeddings
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        pos_embeds = self.pos_embedding(positions)

        # Task conditioning
        task_embed = self.task_embedding(torch.tensor([task_id], device=x.device))
        task_embeds = task_embed.unsqueeze(1).expand(-1, seq_len, -1)

        return token_embeds + pos_embeds + task_embeds

    def _encode_input(self, x: torch.Tensor, task_id: int, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode input through shared transformer."""
        embeds = self._get_embeddings(x, task_id)

        # Apply transformer layers
        for layer in self.transformer_layers:
            embeds = layer(embeds, src_key_padding_mask=mask)

        return embeds

    def forward(
        self,
        x: torch.Tensor,
        task_name: str,
        K: int = 10,
        return_all_steps: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Multi-task forward pass with task-specific reasoning.

        Args:
            x: Input sequence (batch_size, seq_len)
            task_name: Task identifier ('arc', 'sudoku', 'maze')
            K: Number of recursive steps
            return_all_steps: Return predictions at each step

        Returns:
            final_pred: Final prediction (batch_size, vocab_size)
            step_preds: All step predictions if return_all_steps=True
        """
        batch_size = x.size(0)
        task_id = self.get_task_id(task_name)

        # Initialize latent state (task-specific)
        z = self.latent_init[task_id].unsqueeze(0).expand(batch_size, -1)

        # Initialize answer embedding (task-conditioned)
        y = torch.zeros(batch_size, self.embed_dim, device=x.device)
        task_embed = self.task_embedding(torch.tensor([task_id], device=x.device))
        y = y + task_embed

        step_preds = [] if return_all_steps else None

        # Recursive reasoning loop
        for step in range(K):
            # Encode the input sequence first
            x_encoded = self._encode_input(x, task_id)  # (batch_size, seq_len, embed_dim)

            # Use pooled representation of input and current answer for reasoning
            x_pooled = x_encoded.mean(dim=1)  # (batch_size, embed_dim)

            # Update latent state z from (x_pooled, y, z, task_embedding)
            z_input = torch.cat([x_pooled, y, z, task_embed.expand(batch_size, -1)], dim=-1)
            z = self.update_z_net(z_input)

            # Update latent state z from (x_pooled, y, z, task_embedding)
            z_input = torch.cat([x_pooled, y, z, task_embed.expand(batch_size, -1)], dim=-1)
            z = self.update_z_net(z_input)

            # Update answer y from (y, z, task_embedding)
            y_input = torch.cat([y, z, task_embed.expand(batch_size, -1)], dim=-1)
            y = self.update_y_net(y_input)

            # Generate prediction for this step
            step_pred = self.task_heads[task_name](y)

            if return_all_steps:
                step_preds.append(step_pred)

        # Final prediction using task-specific head
        final_pred = self.task_heads[task_name](y)

        return final_pred, step_preds

    def get_loss(
        self,
        x: torch.Tensor,
        targets: torch.Tensor,
        task_name: str,
        K: int = 10,
        supervision_weight: float = 0.1
    ) -> torch.Tensor:
        """
        Compute multi-task loss with optional deep supervision.

        Args:
            x: Input sequences
            targets: Target labels (task-specific format)
            task_name: Task identifier
            K: Recursion depth
            supervision_weight: Weight for intermediate supervision

        Returns:
            loss: Combined loss
        """
        final_pred, step_preds = self.forward(x, task_name, K=K, return_all_steps=True)

        # Task-specific loss functions
        if task_name == 'arc':
            # Cross-entropy for token prediction
            loss_fn = nn.CrossEntropyLoss()
            final_loss = loss_fn(final_pred, targets)

            # Deep supervision on intermediate steps
            supervision_loss = 0.0
            if step_preds:
                for step_pred in step_preds[-3:]:  # Last 3 steps
                    supervision_loss += loss_fn(step_pred, targets)

                supervision_loss /= 3.0
                final_loss = (1 - supervision_weight) * final_loss + supervision_weight * supervision_loss

        elif task_name == 'sudoku':
            # Cross-entropy for digit prediction (1-9, 0=empty)
            loss_fn = nn.CrossEntropyLoss()
            final_loss = loss_fn(final_pred, targets)

        elif task_name == 'maze':
            # Simplified: predict path elements
            loss_fn = nn.CrossEntropyLoss()
            final_loss = loss_fn(final_pred, targets)

        else:
            raise ValueError(f"Unknown task: {task_name}")

        return final_loss


def create_multitask_trm(
    vocab_size: int = 12,
    embed_dim: int = 128,
    latent_dim: int = 64,
    num_layers: int = 2,
    num_heads: int = 8,
    ff_dim: int = 512,
    num_tasks: int = 3,
    dropout: float = 0.1
) -> MultiTaskTRM:
    """Create a multi-task TRM model."""
    return MultiTaskTRM(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        latent_dim=latent_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        ff_dim=ff_dim,
        num_tasks=num_tasks,
        dropout=dropout
    )


# Task-specific configurations (Samsung-style)
TASK_CONFIGS = {
    'arc': {
        'recursion_depth': 10,
        'supervision_weight': 0.1,
        'batch_size': 8
    },
    'sudoku': {
        'recursion_depth': 6,
        'supervision_weight': 0.05,
        'batch_size': 16
    },
    'maze': {
        'recursion_depth': 15,  # Deeper recursion for complex mazes
        'supervision_weight': 0.2,
        'batch_size': 4
    }
}


if __name__ == "__main__":
    # Test the multi-task model
    model = create_multitask_trm()

    # Test with different tasks
    batch_size, seq_len = 4, 64

    # ARC input
    arc_input = torch.randint(0, 11, (batch_size, seq_len))  # 0-10 colors
    arc_output, _ = model(arc_input, 'arc', K=5)
    print(f"ARC output shape: {arc_output.shape}")  # Should be [4, 12]

    # Sudoku input
    sudoku_input = torch.randint(0, 10, (batch_size, seq_len))  # 0-9 digits
    sudoku_output, _ = model(sudoku_input, 'sudoku', K=3)
    print(f"Sudoku output shape: {sudoku_output.shape}")  # Should be [4, 10]

    # Maze input
    maze_input = torch.randint(0, 5, (batch_size, seq_len))  # 0-4 maze elements
    maze_output, _ = model(maze_input, 'maze', K=8)
    print(f"Maze output shape: {maze_output.shape}")  # Should be [4, 5]

    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")