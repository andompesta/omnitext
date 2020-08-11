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
        super(Encoder, self).__init__()
        self.pad_idx = conf.pad_idx
        self.vocab_size = conf.vocab_size
        self.hidden_size = conf.hidden_size
        self.initializer_range = conf.initializer_range
        self.embedding_scale = conf.embedding_scale
        self.max_position_embeddings = conf.max_position_embeddings
        self.learned_pos_embeddings = conf.learned_pos_embeddings
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
            learned=self.learned_pos_embeddings
        )

        self.embed_layer_norm = nn.LayerNorm(
            self.hidden_size,
            eps=conf.layer_norm_eps,
            elementwise_affine=True
        )

        self.dropout = nn.Dropout(conf.hidden_dropout_prob)

        self.layer = nn.ModuleList([EncoderLayer(conf) for _ in range(self.num_hidden_layers)])

    def forward(
            self,
            input_ids: Tensor,
            attention_mask: Tensor,
            output_attentions: bool,
            output_hidden_states: bool,
            position_ids: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor]], Optional[Tuple[Tensor]]]:

        # compute embedding
        hidden_state = self.embed_tokens(input_ids)

        if self.embed_scale is not None:
            hidden_state *= self.embed_scale

        hidden_state += self.embed_positions(input_ids, positions=position_ids)
        hidden_state = self.embed_layer_norm(hidden_state)
        hidden_state = self.dropout(hidden_state)


        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_state,)


            layer_outputs = layer_module(
                hidden_state=hidden_state,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
            )
            hidden_state = layer_outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_state,)

        return (hidden_state, all_hidden_states, all_attentions)