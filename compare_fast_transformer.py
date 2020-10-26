import torch
import numpy as np
from fast_transformers.builders import TransformerEncoderBuilder
from fast_transformers.feature_maps import Favor

from copy import deepcopy
from src.config import XLMRobertaConfig
from src.modules import Encoder
np.random.seed(0)

if __name__ == '__main__':
    x = torch.rand(
        10,    # batch_size
        512,   # sequence length
        64*12  # features
    )

    conf = XLMRobertaConfig(
        num_hidden_layers=3,
        attention_type="linear"
    )
    xlm = Encoder(conf)

    # Build a transformer encoder
    bert = TransformerEncoderBuilder.from_kwargs(
        n_layers=3,
        n_heads=12,
        query_dimensions=64,
        value_dimensions=64,
        feed_forward_dimensions=3072,
        attention_type="linear",
        feature_map=Favor.factory(n_dims=256)
    ).get()

    for xlm_param, bert_param in zip(xlm.parameters(), bert.parameters()):
        xlm_param.data = deepcopy(bert_param.data)

    xlm.eval()
    bert.eval()

    y = bert(x)

    torch.save(
        bert.state_dict(),
        "bert.pt"
    )

    for xlm_l, b_l in zip(xlm.layer,  bert.layers):
        xlm_l.attn.kernel.omega.data = deepcopy(b_l.attention.inner_attention.feature_map.omega.data)

    torch.save(
        xlm.state_dict(),
        "xlm.pt"
    )


    y_ = xlm(x)
    print("bn")
