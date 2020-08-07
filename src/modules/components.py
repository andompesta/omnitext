import torch
import numpy as np
from torch import nn
from typing import Optional, Tuple

from .utils import ACT_NAME_TO_FN, create_sinusoidal_embeddings, create_position_ids_from_input_ids
from src.config import XLMRobertaConfig


class CombinedEmbedding(nn.Module):
    """
    Construct the embedding from word and position
    """
    def __init__(
            self,
            conf: XLMRobertaConfig,
            freeze: bool = True,
            is_sinusoidal: bool = True,
            scale: Optional[float] = 1.
    ):
        super().__init__()
        self.vocab_size = conf.vocab_size
        self.hidden_size = conf.hidden_size
        self.max_position_embeddings = conf.max_position_embeddings
        self.initializer_range = conf.initializer_range
        self.layer_norm_eps = conf.layer_norm_eps
        self.hidden_dropout_prob = conf.hidden_dropout_prob
        self.pad_idx = conf.pad_idx

        self.scale = scale
        self.freeze = freeze
        self.is_sinusoidal = is_sinusoidal

        self.word_embeddings = nn.Embedding(self.vocab_size,
                                            self.hidden_size,
                                            padding_idx=self.pad_idx)
        self.position_embeddings = nn.Embedding(conf.max_position_embeddings,
                                                conf.hidden_size)
        self.LayerNorm = nn.LayerNorm(self.hidden_size,
                                      eps=conf.layer_norm_eps,
                                      elementwise_affine=True)
        self.dropout = nn.Dropout(p=self.hidden_dropout_prob)

    def _init_weights(self) -> None:
        self.word_embeddings.weight.data.normal_(mean=.0,
                                                 std=self.initializer_range)

        if self.pad_idx is not None:
            nn.init.constant_(self.word_embeddings.weight[self.pad_idx], 0)

        if self.is_sinusoidal:
            create_sinusoidal_embeddings(self.max_position_embeddings, self.hidden_size,
                                         out=self.position_embeddings.weight)
        else:
            self.position_embeddings.weight.data.normal_(mean=.0,
                                                         std=self.initializer_range)

    def forward(
            self,
            input_ids: torch.Tensor,
            position_ids: torch.Tensor
    ) -> torch.Tensor:

        embedding = self.word_embeddings(input_ids)

        if self.scale is not None:
            embedding += self.scale

        embedding += self.position_embeddings(position_ids)

        embedding = self.LayerNorm(embedding)
        embedding = self.dropout(embedding)
        return embedding


class SelfAttention(nn.Module):
    def __init__(self, conf: XLMRobertaConfig):
        super().__init__()
        self.output_attentions = conf.output_attentions
        self.num_attention_heads = conf.num_attention_heads
        self.attention_head_size = int(conf.hidden_size / conf.num_attention_heads)

    def forward(self):
        return