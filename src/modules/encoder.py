from torch import nn, Tensor
from typing import Optional, Tuple

from src.config import XLMRobertaConfig
from src.modules import (
    TokenEmbedding,
    PositionalEmbedding,
    EncoderLayer
)

class Encoder(nn.Module):
    def __init__(self, conf: XLMRobertaConfig):
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
            self.vocab_size, self.hidden_size, self.pad_idx, self.initializer_range, self.embedding_scale
        )

        self.embed_positions = PositionalEmbedding(
            self.max_position_embeddings, self.hidden_size, self.pad_idx,
            initializer_range=self.initializer_range, learned=self.learned_pos_embeddings
        )

        self.emb_layer_norm = nn.LayerNorm(
            self.hidden_size,
            eps=conf.layer_norm_eps, elementwise_affine=True
        )

        self.dropout = nn.Dropout(conf.hidden_dropout_prob)

        self.layer = nn.ModuleList([EncoderLayer(conf) for _ in range(self.num_hidden_layers)])

    def forward(
            self,
            hidden_state: Tensor,
            attention_mask: Optional[Tensor] = None,
            output_attentions=False,
            output_hidden_states=False,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor]], Optional[Tuple[Tensor]]]:
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