from typing import Dict, Optional
import torch
import torch.nn.functional as F
from torch import Tensor
from .utils import create_position_ids_from_input_ids


class PartiallyFixedLearnedPositionalEmbedding(torch.nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    Padding ids are ignored by either offsetting based on padding_idx
    or by setting padding_idx to None and ensuring that the appropriate
    position ids are passed to the forward function.
    """

    def __init__(
            self,
            num_embeddings: int,
            embedding_dim: int,
            padding_idx: int,
            initializer_range: float,
            trainable_num_embeddings: int,
            fixed_num_embeddings: int
    ):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.fixed_num_embeddings = fixed_num_embeddings
        self.trainable_num_embeddings = trainable_num_embeddings
        self.initializer_range = initializer_range
        self.max_positions = self.num_embeddings
        self.register_buffer(
            "learnable_mask", (torch.arange(self.num_embeddings).unsqueeze(1) >= self.fixed_num_embeddings).float()
        )

    def forward(
        self,
        input_ids: Tensor,
        positions: Optional[Tensor] = None,
    ):
        """Input is expected to be of size [bsz x seqlen]."""
        assert (positions is None) or (
            self.padding_idx is None
        ), "If positions is pre-computed then padding_idx should not be set."

        if positions is None:
            positions = create_position_ids_from_input_ids(input_ids, self.padding_idx)

        assert positions.max() < self.max_positions

        self.weight.register_hook(self._zero_grad_fixed)

        return F.embedding(
            positions,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )

    def _zero_grad_fixed(
            self,
            gr: torch.Tensor
    ) -> torch.Tensor:
        return gr * self.learnable_mask

    def _init_weights(self) -> None:
        self.weight.data.normal_(mean=.0, std=self.initializer_range)
        if self.padding_idx is not None:
            torch.nn.init.constant_(self.weight.data[self.padding_idx], 0)
