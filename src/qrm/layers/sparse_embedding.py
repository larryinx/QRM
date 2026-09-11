"""Sparse Embedding layer for puzzle embeddings."""

import logging
import os
from typing import Optional

import torch
from torch import nn

from qrm.layers.common import trunc_normal_init_

logger = logging.getLogger(__name__)


class CastedSparseEmbedding(nn.Module):
    """Sparse embedding layer with optional low-rank factorization.

    Key design points:
    1. weights stored as Buffer (not Parameter)
    2. local_weights has gradients for backpropagation
    3. local_ids records current batch IDs for SignSGD

    Low-rank mode (when rank is specified):
    - weights: [num_embeddings, rank] instead of [num_embeddings, embedding_dim]
    - proj: nn.Linear(rank, embedding_dim) as Parameter, optimized by Adam
    - Output: proj(weights[inputs])
    - SignSGD only updates weights, Adam updates proj
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        batch_size: int,  # Required: need to pre-allocate local_weights
        init_std: float,
        cast_to: torch.dtype,
        rank: Optional[int] = None,  # Low-rank dimension, None = full rank
    ):
        super().__init__()
        self.cast_to = cast_to
        self.rank = rank
        self.embedding_dim = embedding_dim

        # Internal dimension: use rank if specified, otherwise full embedding_dim
        internal_dim = rank if rank is not None else embedding_dim

        # Real Weights with Truncated LeCun normal init
        self.weights = nn.Buffer(
            trunc_normal_init_(
                torch.empty((num_embeddings, internal_dim)), std=init_std
            ),
            persistent=True,
        )

        # Local weights and IDs
        # Local embeddings, with gradient, not persistent
        self.local_weights = nn.Buffer(
            torch.zeros(batch_size, internal_dim, requires_grad=True), persistent=False
        )
        # Local embedding IDs, not persistent
        self.local_ids = nn.Buffer(
            torch.zeros(batch_size, dtype=torch.int32), persistent=False
        )

        # Low-rank projection layer (Parameter, optimized by Adam)
        if rank is not None:
            self.proj = nn.Linear(rank, embedding_dim, bias=False)
        else:
            self.proj = None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        if not self.training:
            # Test mode, no gradient
            emb = self.weights[inputs]
            if self.proj is not None:
                emb = self.proj(emb)
            return emb.to(self.cast_to)

        # Training mode, fill puzzle embedding from weights
        current_batch_size = inputs.shape[0]
        buffer_batch_size = self.local_weights.shape[0]

        # Handle batch size changes (e.g., epoch boundaries)
        # Strategy: don't resize buffer, only use first N elements
        # This keeps SignSGD optimizer param_groups references valid
        # Remaining local_ids keep old values but have zero gradients, not affecting SignSGD
        if current_batch_size < buffer_batch_size:
            gpu_rank = int(os.environ.get("RANK", 0))
            logger.warning(
                f"[Rank {gpu_rank}] CastedSparseEmbedding batch size mismatch: "
                f"current={current_batch_size}, buffer={buffer_batch_size}. "
                f"Using first {current_batch_size} elements of buffer."
            )

        if current_batch_size > buffer_batch_size:
            raise RuntimeError(
                f"Batch size {current_batch_size} exceeds buffer size {buffer_batch_size}. "
                f"This should not happen if model is initialized with the correct batch_size."
            )

        with torch.no_grad():
            self.local_weights[:current_batch_size].copy_(self.weights[inputs])
            self.local_ids[:current_batch_size].copy_(inputs)

        emb = self.local_weights[:current_batch_size]
        if self.proj is not None:
            emb = self.proj(emb)
        return emb.to(self.cast_to)
