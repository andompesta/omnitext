import copy
from src.utils import ensure_dir, save_data_to_json, load_data_from_json
from abc import ABC, abstractmethod

class BaseConf(ABC):
    def __init__(self, d_model: int, n_head: int, n_layer: int):
        self.d_model = d_model
        self.n_head = n_head
        self.n_layer = n_layer

    @abstractmethod
    @property
    def name(self) -> str:
        ...

    @property
    def hidden_size(self):
        return self.d_model

    @property
    def num_attention_heads(self):
        return self.n_head

    @property
    def num_hidden_layers(self):
        return self.n_layer

    def to_dict(self) -> dict:
        output = copy.deepcopy(self.__dict__)
        return output

    def save(self, path_: str):
        save_data_to_json(ensure_dir(path_), self.to_dict())

    @classmethod
    def from_dict(cls, json_object: dict) -> object:
        return cls(**json_object)

    @classmethod
    def load(cls, path_: str):
        json_obj = load_data_from_json(path_)
        return cls(**json_obj)