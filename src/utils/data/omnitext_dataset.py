
import numpy as np
from torch import LongTensor
from torch.utils.data import Dataset
from abc import abstractmethod
from typing import List

class OmniDataset(Dataset):
    """
    abstract cals for my datasets
    """

    _tensor_type = LongTensor

    @property
    def tensor_type(self):
        """
        ensure all dataset has the same tensor type
        """
        return OmniDataset._tensor_type

    @property
    def support_prefetch(self):
        """
        Wheter this dataset support prefetch
        """
        return False


    def __init__(self, pad_token_id: int, eos_token_id: int, shuffle: bool = True):
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.shuffle = shuffle
        self.epoch = 1

    def __getitem__(self, item):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    @abstractmethod
    def collate(self, samples: List):
        """
        collate function to apply to a list of samples
        :param samples:
        :return:
        """
        ...

    @abstractmethod
    def num_tokens(self, index: int):
        """
        return the number of token in a sample. This method is used to enforce `--max-tokens`` during batching
        :param index: index of the sample
        :return:
        """
        ...

    def ordered_indices(self) -> List[int]:
        """
        Return an ordered list of indices. Batches are constructed based on this order.
        :return: array of index
        """
        return list(range(len(self)))

    @abstractmethod
    def prefetch(self, indices: List[int]):
        """
        Prefetch the data for this epoch
        :param indeces:
        :return:
        """
        ...

    def set_epoch(self, epoch):
        self.epoch = epoch