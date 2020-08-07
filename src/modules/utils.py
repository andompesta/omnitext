import torch
import math
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional
from os import path

from src.config import BaseConf
from src.utils import ensure_dir
from shutil import copyfile

class BaseModel(torch.nn.Module):
    def __init__(self, conf: BaseConf):
        super(BaseModel, self).__init__()
        self.conf = conf
        self.name = conf.name

    def init_weights(self) -> None:
        """Initialize weights if needed."""
        self.apply(self._init_weights)

    @abstractmethod
    def _init_weights(self, module: torch.nn.Module):
        """Child model has to define the initialization policy."""
        ...

    @abstractmethod
    def get_output_embeddings(self) -> torch.nn.Module:
        """Returns model output embeddings"""
        ...

    @abstractmethod
    def get_input_embeddings(self) -> torch.nn.Module:
        """Returns model output embeddings"""
        ...

    def tie_weights(self) -> None:
        """
        Tie weights between input and output embeddings
        :return:
        """
        out_embeddings = self.get_output_embeddings()
        if out_embeddings is not None:
            in_embeddings = self.get_input_embeddings()
            self._tie_or_clone_weights(out_embeddings, in_embeddings)

    def _tie_or_clone_weights(self, out_embeddings: torch.nn.Module, in_embeddings: torch.nn.Module):
        out_embeddings.weight = in_embeddings.weight
        if hasattr(out_embeddings, "bias") and out_embeddings.bias is not None:
            out_embeddings.bias.data = torch.nn.functional.pad(
                out_embeddings.bias.data,
                (0, out_embeddings.weight.shape[0] - out_embeddings.bias.shape[0]),
                "constant",
                0
            )

        if hasattr(out_embeddings, "out_features") and hasattr(in_embeddings, "num_embeddings"):
            out_embeddings.out_features = in_embeddings.num_embeddings

    def save(self, path_: str, is_best: Optional[bool], file_name='checkpoint.pth.tar') -> None:
        if isinstance(self, torch.nn.DataParallel):
            state_dict = dict([(key, value.to("cpu")) for key, value in self.module.state_dict().items()])
        else:
            state_dict = dict([(key, value.to("cpu")) for key, value in self.state_dict().items()])

            torch.save(state_dict, ensure_dir(path.join(path_, file_name)))
            if is_best:
                copyfile(path.join(path_, file_name), path.join(path_, "model_best.pth.tar"))

    @classmethod
    def load(cls, conf: BaseConf, path_: str, mode: str = 'trained'):
        model = cls(conf)
        state_dict = torch.load(path_, map_location="cpu")

        if mode == 'pre-trained':
            strict = False
        elif mode == 'trained':
            state_dict = state_dict['state_dict']
            strict = True
        else:
            raise NotImplementedError()

        model.load_state_dict(state_dict, strict=strict)
        return model



def create_position_ids_from_input_ids(
        input_ids: torch.Tensor,
        padding_idx: int
) -> torch.Tensor:
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at
    padding_idx+1. Padding symbols are ignored. This is modified from fairseq's

    :param input_ids: tokens ids
    :param padding_idx: Pad index
    :return: positional embedding ids
    """
    # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = torch.cumsum(mask, dim=1).type_as(mask) * mask
    return incremental_indices.long() + padding_idx

def create_sinusoidal_embeddings(
        n_pos: int,
        dim: int,
        out: torch.Tensor
):
    """
    Create a sinusoidal embedding for the positional embedding.
    The embedding is detached thus with no gradient.

    :param n_pos: max sequence length
    :param dim: embedding dimension
    :param out: output tensor
    :return:
    """

    position_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)
    ])
    out[:, 0::2] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
    out[:, 1::2] = torch.FloatTensor(np.cos(position_enc[:, 0::2]))
    out.detach_()
    out.requires_grad = False


def swich (x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)

def gelu_accurate(x: torch.Tensor) -> torch.Tensor:
    if not hasattr(gelu_accurate, "_a"):
        gelu_accurate._a = math.sqrt(2 / math.pi)
    return (
        0.5 * x * (1 + torch.tanh(gelu_accurate._a * (x + 0.044715 * torch.pow(x, 3))))
    )

def gelu(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.gelu(x.float()).type_as(x)


ACT_NAME_TO_FN = dict(gelu=gelu,
                      gelu_accurate=gelu_accurate,
                      swich=swich)