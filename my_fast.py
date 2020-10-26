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
    xlm.eval()
    y_ = xlm(x)
    print("bn")
