from torch import nn, Tensor
from typing import Optional

from src.config import BaseConf
from src.modules import (
    EncoderLayer
)

class Encoder(nn.Module):
    def __init__(self, conf: BaseConf):
        super(Encoder, self).__init__()
        self.layer = nn.ModuleList([EncoderLayer(conf) for _ in range(conf.num_hidden_layers)])

    def forward(
            self,
            hidden_states: Tensor,
            attention_mask: Optional[Tensor] = None,
            **kwargs
    ) -> Tensor:
        for i, layer_module in enumerate(self.layer):
            hidden_states = layer_module(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                **kwargs
            )
        return hidden_states
