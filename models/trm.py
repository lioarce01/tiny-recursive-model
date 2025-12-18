"""
Tiny Recursive Model (TRM) implementation based on Samsung's "Less is More" paper.

This model implements a recursive reasoning architecture with a 2-layer transformer
that performs iterative latent state updates for reasoning tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional


class TransformerBlock(nn.Module):
    """Single transformer block with self-attention and feed-forward layers."""

    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual connection
        attn_output, _ = self.attention(x, x, x, key_padding_mask=mask)
        x = self.norm1(x + attn_output)

        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)

        return x


class TinyRecursiveModel(nn.Module):
    """
    Tiny Recursive Model for reasoning tasks.

    Architecture:
    - 2-layer transformer with learned positional embeddings
    - Recursive loop with K steps: update latent z from (x,y,z), update y from (y,z)
    - Deep supervision across recursion steps
    """

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 1024,
        embed_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        ff_dim: int = 256,
        latent_dim: int = 32,
        dropout: float = 0.3
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim

        # Token embeddings and positional embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)

        # 2-layer transformer
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

        # Latent state update networks
        # Update z from (x, y, z) - combine input, current answer, and latent state
        self.update_z_net = nn.Sequential(
            nn.Linear(embed_dim + embed_dim + latent_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, latent_dim)
        )

        # Update y from (y, z) - refine answer using latent state
        self.update_y_net = nn.Sequential(
            nn.Linear(embed_dim + latent_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )

        # Final prediction head (for classification/regression tasks)
        self.predict_head = nn.Linear(embed_dim, vocab_size)

        # Initialize latent state
        self.latent_init = nn.Parameter(torch.randn(latent_dim))

    def _get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Get token and positional embeddings for input sequence."""
        seq_len = x.size(1)

        # Token embeddings
        token_embeds = self.token_embedding(x)

        # Positional embeddings (learned)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        pos_embeds = self.pos_embedding(positions)

        return token_embeds + pos_embeds

    def _encode_input(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode input sequence through transformer layers."""
        # Get embeddings
        embeds = self._get_embeddings(x)

        # Apply transformer layers
        for layer in self.transformer_layers:
            embeds = layer(embeds, mask)

        return embeds

    def forward(
        self,
        x: torch.Tensor,
        y_init: Optional[torch.Tensor] = None,
        K: int = 10,
        return_all_steps: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with recursive reasoning.

        Args:
            x: Input sequence (batch_size, seq_len)
            y_init: Initial answer embedding (batch_size, embed_dim), if None uses zeros
            K: Number of recursive steps
            return_all_steps: Whether to return predictions at all steps for supervision

        Returns:
            final_pred: Final prediction logits (batch_size, vocab_size)
            step_preds: Predictions at each step (K, batch_size, vocab_size) if return_all_steps
        """
        batch_size = x.size(0)

        # Encode input once
        x_encoded = self._encode_input(x)  # (batch_size, seq_len, embed_dim)

        # Initialize latent state and answer
        z = self.latent_init.unsqueeze(0).expand(batch_size, -1)  # (batch_size, latent_dim)

        if y_init is None:
            y = torch.zeros(batch_size, self.embed_dim, device=x.device)
        else:
            y = y_init

        step_preds = []

        # Recursive loop for K steps
        for step in range(K):
            # Step 1: Update latent state z from (x, y, z)
            # Aggregate x_encoded (mean pooling across sequence dimension)
            x_agg = x_encoded.mean(dim=1)  # (batch_size, embed_dim)

            # Concatenate x, y, z for latent update
            z_input = torch.cat([x_agg, y, z], dim=-1)  # (batch_size, embed_dim + embed_dim + latent_dim)
            z = self.update_z_net(z_input)  # (batch_size, latent_dim)

            # Step 2: Update answer y from (y, z)
            y_input = torch.cat([y, z], dim=-1)  # (batch_size, embed_dim + latent_dim)
            y = self.update_y_net(y_input)  # (batch_size, embed_dim)

            # Generate prediction at this step
            step_pred = self.predict_head(y)  # (batch_size, vocab_size)
            step_preds.append(step_pred)

        # Final prediction is the last step's prediction
        final_pred = step_preds[-1]

        if return_all_steps:
            # Stack predictions from all steps
            step_preds = torch.stack(step_preds, dim=0)  # (K, batch_size, vocab_size)
            return final_pred, step_preds
        else:
            return final_pred, None

    def get_loss(
        self,
        x: torch.Tensor,
        targets: torch.Tensor,
        K: int = 10,
        supervision_weight: float = 0.1
    ) -> torch.Tensor:
        """
        Compute loss with deep supervision across recursion steps.

        Args:
            x: Input sequences (batch_size, seq_len)
            targets: Target labels (batch_size,)
            K: Number of recursive steps
            supervision_weight: Weight for intermediate supervision loss

        Returns:
            Total loss combining final and intermediate predictions
        """
        final_pred, step_preds = self.forward(x, K=K, return_all_steps=True)

        # Final step loss
        final_loss = F.cross_entropy(final_pred, targets)

        # Deep supervision loss (average across all intermediate steps)
        if step_preds is not None and supervision_weight > 0:
            step_losses = []
            for step_pred in step_preds[:-1]:  # Exclude final step (already in final_loss)
                step_loss = F.cross_entropy(step_pred, targets)
                step_losses.append(step_loss)
            supervision_loss = torch.stack(step_losses).mean() if step_losses else 0.0
        else:
            supervision_loss = 0.0

        total_loss = final_loss + supervision_weight * supervision_loss
        return total_loss


def create_trm_model(
    vocab_size: int,
    max_seq_len: int = 1024,
    embed_dim: int = 128,
    **kwargs
) -> TinyRecursiveModel:
    """Factory function to create TRM model with default hyperparameters."""
    return TinyRecursiveModel(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        embed_dim=embed_dim,
        **kwargs
    )