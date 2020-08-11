import torch
import math
from typing import Optional
from abc import abstractmethod
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

    def save(self, path_: str, is_best: Optional[bool] = False, file_name:str = 'checkpoint.pth.tar') -> None:
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