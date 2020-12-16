import torch
from torch import nn

from src.config import BaseConf
from .utils import ACT_NAME_TO_FN

class LMHead(nn.Module):
    """Head for masked language modeling."""

    def __init__(self, conf: BaseConf):
        super().__init__()
        self.hidden_size = conf.hidden_size
        self.layer_norm_eps = conf.layer_norm_eps
        self.output_dim = conf.vocab_size

        self.dense = nn.Linear(self.hidden_size, self.hidden_size)
        self.activation_fn = ACT_NAME_TO_FN[conf.hidden_act]
        self.layer_norm = nn.LayerNorm(
            self.hidden_size,
            eps=self.layer_norm_eps,
            elementwise_affine=True
        )

        self.decoder = nn.Linear(self.hidden_size, self.output_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(self.output_dim))
        self.decoder.bias = self.bias

    def forward(
            self,
            features: torch.Tensor,
    ) -> torch.Tensor:
        x = self.dense(features)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        # project back to size of vocabulary with bias
        x = self.decoder(x)
        return x

class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, conf: BaseConf):
        super().__init__()
        self.hidden_size = conf.hidden_size
        self.num_labels = conf.num_labels

        self.dropout = nn.Dropout(p=conf.hidden_dropout_prob)
        self.out_proj = nn.Linear(self.hidden_size, self.num_labels)

    def forward(
            self,
            hidden_state: torch.Tensor
    ) -> torch.Tensor:
        hidden_state = self.dropout(hidden_state)
        hidden_state = self.out_proj(hidden_state)
        return hidden_state
