from torch import nn, Tensor
from typing import Optional, Tuple

from src.config import BaseConf
from src.modules import (
    SelfOutput,
    SelfAttention,
    Intermediate,
    Output
)

class EncoderLayer(nn.Module):
    def __init__(self, conf: BaseConf):
        super(EncoderLayer, self).__init__()
        self.self_attn = SelfAttention(conf)
        self.self_output = SelfOutput(conf)
        self.intermediate = Intermediate(conf)
        self.output = Output(conf)

    def forward(
            self,
            hidden_state: Tensor,
            attention_mask: Optional[Tensor] = None,
            output_attentions: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        residual = hidden_state
        self_attention_outputs = self.self_attn(hidden_state,
                                                attention_mask=attention_mask,
                                                output_attentions=output_attentions)
        hidden_state = self_attention_outputs[0]
        hidden_state = self.self_output(hidden_state, residual)
        outputs = self_attention_outputs[1:]

        residual = hidden_state
        hidden_state = self.intermediate(hidden_state)
        hidden_state = self.output(hidden_state, residual)
        outputs = (hidden_state, ) + outputs
        return outputs

