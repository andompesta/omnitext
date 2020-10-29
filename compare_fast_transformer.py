import torch
import numpy as np
from fast_transformers.builders import TransformerEncoderBuilder
from fast_transformers.feature_maps import Favor

from copy import deepcopy
from src.config import XLMRobertaConfig
from src.modules import Encoder
np.random.seed(0)

if __name__ == '__main__':
