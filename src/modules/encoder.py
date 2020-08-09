import torch
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
            input_ids: Tensor,
            position_ids: Optional[Tensor] = None,
            attention_mask: Optional[Tensor] = None,
            output_attentions=False,
            output_hidden_states=False,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor]], Optional[Tuple[Tensor]]]:

        if attention_mask is None:
            # do not attend of pad_idx
            attention_mask = input_ids.eq(self.pad_idx)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask)

        # compute embedding
        hidden_state = self.embed_tokens(input_ids)

        if self.embed_scale is not None:
            hidden_state *= self.embed_scale

        hidden_state += self.embed_positions(input_ids, positions=position_ids)
        hidden_state = self.emb_layer_norm(hidden_state)
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

    def get_extended_attention_mask(
            self,
            attention_mask: Tensor
    ) -> Tensor:
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.
        Arguments:
            attention_mask (:obj:`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
        Returns:
            :obj:`torch.Tensor` The extended attention mask, with a the same dtype as :obj:`attention_mask.dtype`.
        """
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                "Wrong shape for attention_mask (shape {})".format(attention_mask.shape)
            )

        # Since attention_mask is 0.0 for positions we want to attend and 1.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        extended_attention_mask = extended_attention_mask * -10000.0
        return extended_attention_mask