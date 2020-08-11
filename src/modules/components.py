import torch
from torch import nn, Tensor
from typing import Optional, Tuple

from .utils import ACT_NAME_TO_FN
from src.config import BaseConf


class SelfAttention(nn.Module):
    def __init__(self, conf: BaseConf):
        super().__init__()
        if conf.hidden_size % conf.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention head (%d)" %
                (conf.hidden_size, conf.num_attention_heads)
            )

        self.hidden_size = conf.hidden_size
        self.num_attention_heads = conf.num_attention_heads
        self.attention_head_size = int(conf.hidden_size / conf.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(self.hidden_size, self.all_head_size)
        self.key = nn.Linear(self.hidden_size, self.all_head_size)
        self.value = nn.Linear(self.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(conf.attention_probs_dropout_prob)
        self.scaling = self.attention_head_size ** -0.5

    def transpose_for_scores(self, x: Tensor) -> Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self. attention_head_size)
        x = x.view(*new_x_shape)
        x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def forward(self,
                hidden_state: Tensor,
                attention_mask: Optional[Tensor] = None,
                encoder_hidden_states: Optional[Tensor] = None,
                encoder_attention_mask: Optional[Tensor] = None,
                output_attentions: bool = False
                ) -> Tuple[Tensor, Optional[Tensor]]:
        query_h = self.query(hidden_state)

        if encoder_hidden_states is not None:
            key_h = self.key(encoder_hidden_states)
            value_h = self.value(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        else:
            key_h = self.key(hidden_state)
            value_h = self.value(hidden_state)

        query_h = self.transpose_for_scores(query_h)
        key_h = self.transpose_for_scores(key_h)
        value_h = self.transpose_for_scores(value_h)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_h, key_h.transpose(-1, -2))
        attention_scores = attention_scores * self.scaling
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        context_h = torch.matmul(attention_probs, value_h)
        context_h = context_h.permute(0, 2, 1, 3).contiguous()
        next_context_h_shape = context_h.size()[:-2] + (self.all_head_size,)
        context_h = context_h.view(*next_context_h_shape)

        output = (context_h, attention_probs) if output_attentions else (context_h, )
        return output

class SelfOutput(nn.Module):
    def __init__(self, conf: BaseConf):
        super().__init__()

        self.hidden_size = conf.hidden_size
        self.hidden_dropout_prob = conf.hidden_dropout_prob
        self.layer_norm_eps = conf.layer_norm_eps

        self.dense = nn.Linear(self.hidden_size, self.hidden_size)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.layer_norm = nn.LayerNorm(
            self.hidden_size,
            eps=self.layer_norm_eps,
            elementwise_affine=True
        )

    def forward(self, hidden_state: Tensor, residual: Tensor):
        hidden_state = self.dense(hidden_state)
        hidden_state = self.dropout(hidden_state)
        hidden_state = hidden_state + residual
        return self.layer_norm(hidden_state)

class Intermediate(nn.Module):
    def __init__(self, conf: BaseConf):
        super().__init__()
        self.hidden_size = conf.hidden_size
        self.output_size = conf.intermediate_size
        self.dropout_prob = conf.intermediate_dropout_prob

        self.dropout = nn.Dropout(self.dropout_prob)
        self.activation_fn = ACT_NAME_TO_FN[conf.hidden_act]
        self.dense = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, hidden_state: Tensor) -> Tensor:
        hidden_state = self.dense(hidden_state)
        hidden_state = self.activation_fn(hidden_state)
        hidden_state = self.dropout(hidden_state)
        return hidden_state

class Output(nn.Module):
    def __init__(self, conf: BaseConf):
        super(Output, self).__init__()

        self.hidden_size = conf.hidden_size
        self.intermediate_size = conf.intermediate_size
        self.layer_norm_eps = conf.layer_norm_eps
        self.dropout_prob = conf.hidden_dropout_prob

        self.dense = nn.Linear(self.intermediate_size, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_prob)
        self.layer_norm = nn.LayerNorm(self.hidden_size,
                                       eps=self.layer_norm_eps,
                                       elementwise_affine=True)

    def forward(
            self,
            hidden_state: Tensor,
            residual: Tensor
            ) -> Tensor:
        hidden_state = self.dense(hidden_state)
        hidden_state = self.dropout(hidden_state)
        hidden_state = residual + hidden_state
        hidden_state = self.layer_norm(hidden_state)
        return hidden_state
