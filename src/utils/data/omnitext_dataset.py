
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


    def __init__(self, pad: int, shuffle: bool = True):
        self.pad = pad
        self.shuffle = shuffle

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

    @abstractmethod
    def size(self, index: int):
        """
        return an example's size as a float or tuple. This method is used to enforce ``--max-position`` during batching
        :param index:
        :return:
        """
        ...

    def ordered_indices(self) -> List[int]:
        """
        Return an ordered list of indices. Batches are constructed based on this order.
        :return: array of index
        """
        return np.arange(len(self)).tolist()

    @abstractmethod
    def prefetch(self, indices: List[int]):
        """
        Prefetch the data for this epoch
        :param indeces:
        :return:
        """
        ...