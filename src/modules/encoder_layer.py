from torch import nn, Tensor
from typing import Optional, Tuple

from src.config import BaseConf
from src.modules import (
    LinearAttention,
    SelfOutput,
    SelfAttention,
    Intermediate,
    Output
)

class EncoderLayer(nn.Module):
    def __init__(self, conf: BaseConf):
        super(EncoderLayer, self).__init__()
        if conf.attention_type == "full":
            self.attn = SelfAttention(conf)
        elif conf.attention_type == "linear":
            self.attn = LinearAttention(conf)
        else:
            raise NotImplementedError('Not implemented yet. Support only "full" and "linear"')

        self.self_output = SelfOutput(conf)
        self.intermediate = Intermediate(conf)
        self.output = Output(conf)

    def forward(
            self,
            hidden_states: Tensor,
            attention_mask: Optional[Tensor] = None,
            **kwargs
    ) -> Tensor:
        residual = hidden_states
        hidden_states = self.attn(
            hidden_states,
            attention_mask=attention_mask,
            **kwargs
        )

        hidden_states = self.self_output(hidden_states, residual)

        residual = hidden_states
        hidden_states = self.intermediate(hidden_states)
        hidden_states = self.output(hidden_states, residual)
        return hidden_states

