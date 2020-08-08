import math
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F
from .utils import create_position_ids_from_input_ids


class SinusoidalPositionalEmbedding(nn.Embedding):
    """This module produces sinusoidal positional embeddings of any length.
    Padding symbols are ignored.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.max_positions = self.num_embeddings

    def _init_weights(self):
        """Build sinusoidal embeddings.
        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = self.embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(self.num_embeddings, dtype=torch.float).unsqueeze(
            1
        ) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(
            self.num_embeddings, -1
        )
        if self.embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(self.num_embeddings, 1)], dim=1)

        if self.padding_idx is not None:
            emb[self.padding_idx, :] = 0

        emb = emb.detach_()
        emb.requires_grad = False
        self.weights = emb

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
    ):
        """Input is expected to be of size [bsz x seqlen]."""
        assert (positions is None) or (
            self.padding_idx is None
        ), "If positions is pre-computed then padding_idx should not be set."

        if positions is None:
            positions = create_position_ids_from_input_ids(input_ids, self.padding_idx)

        assert positions.max() < self.max_positions

        return F.embedding(
            positions,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )