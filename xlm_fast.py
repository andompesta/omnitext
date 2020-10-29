import torch
import numpy as np
from src.config import XLMRobertaConfig
from src.modules import Encoder
np.random.seed(0)
torch.manual_seed(0)

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

    # xlm.load_state_dict(torch.load("xlm.pt"))
    xlm.eval()
    y = xlm(x)
    print("bn")