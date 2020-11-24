from torch import nn, Tensor
from typing import Optional, Tuple

from src.config import BaseConf
from src.modules import (
    TokenEmbedding,
    PositionalEmbedding,
    EncoderLayer
)

class Encoder(nn.Module):
    def __init__(self, conf: BaseConf):
        super(Encoder, self).__init__(conf)
        self.pad_idx = conf.pad_idx
        self.vocab_size = conf.vocab_size
        self.hidden_size = conf.hidden_size
        self.initializer_range = conf.initializer_range
        self.embedding_scale = conf.embedding_scale
        self.max_position_embeddings = conf.max_position_embeddings
        self.pos_embeddings_type = conf.pos_embeddings_type
        self.num_hidden_layers = conf.num_hidden_layers

        self.embed_tokens = TokenEmbedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.hidden_size,
            padding_idx=self.pad_idx,
            initializer_range=self.initializer_range,
            scale=self.embedding_scale
        )

        self.embed_positions = PositionalEmbedding(
            num_embeddings=self.max_position_embeddings,
            embedding_dim=self.hidden_size,
            padding_idx=self.pad_idx,
            initializer_range=self.initializer_range,
            type=self.pos_embeddings_type
        )

        self.embed_layer_norm = nn.LayerNorm(
            self.hidden_size,
            eps=conf.layer_norm_eps,
            elementwise_affine=True
        )

        self.embed_dropout = nn.Dropout(conf.hidden_dropout_prob)

        self.layer = nn.ModuleList([EncoderLayer(conf) for _ in range(self.num_hidden_layers)])

    def forward(
            self,
            input_ids: Tensor,
            attention_mask: Tensor,
            position_ids: Optional[Tensor] = None,
            **kwargs
    ) -> Tensor:

        # compute embedding
        hidden_state = self.embed_tokens(input_ids)

        if self.embed_scale is not None:
            hidden_state *= self.embed_scale

        hidden_state += self.embed_positions(input_ids, positions=position_ids)
        hidden_state = self.embed_layer_norm(hidden_state)
        hidden_state = self.embed_dropout(hidden_state)

        for i, layer_module in enumerate(self.layer):
            hidden_state = layer_module(
                hidden_state=hidden_state,
                attention_mask=attention_mask,
                **kwargs
            )
        return hidden_state