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
            hidden_state: Tensor,
            attention_mask: Optional[Tensor] = None,
            **kwargs
    ) -> Tensor:
        residual = hidden_state
        hidden_state = self.attn(
            hidden_state,
            attention_mask=attention_mask,
            **kwargs
        )

        hidden_state = self.self_output(hidden_state, residual)

        residual = hidden_state
        hidden_state = self.intermediate(hidden_state)
        hidden_state = self.output(hidden_state, residual)
        outputs = hidden_state
        return outputs

