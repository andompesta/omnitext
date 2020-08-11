# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Optional

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from .utils import create_position_ids_from_input_ids

class LearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    Padding ids are ignored by either offsetting based on padding_idx
    or by setting padding_idx to None and ensuring that the appropriate
    position ids are passed to the forward function.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int, initializer_range: float):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.initializer_range = initializer_range
        self.max_positions = self.num_embeddings

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

        return F.embedding(
            positions,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )

    def _init_weights(self) -> None:
        self.weight.data.normal_(mean=.0, std=self.initializer_range)
        if self.padding_idx is not None:
            nn.init.constant_(self.weight.data[self.padding_idx], 0)