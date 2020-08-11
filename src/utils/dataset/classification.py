import numpy as np
import torch
from typing import Tuple, List, Dict

from src.utils import load_obj_from_file
from src.utils.data import OmniDataset
from src.utils.data import data_utils
from src.utils.data import iterators


def classify_pad_bath(ids: List[List[int]], sizes: List[int], pad: int) -> List[np.array]:
    """
    Pad the instances to max length in the batch
    :param ids: sequences to pad
    :param sizes: size of each sequence
    :param pad: pad index
    :return:
    """

    max_len = max(sizes)
    collector = []
    for seq, size in zip(ids, sizes):
        seq = np.array(seq + [pad] * (max_len - size))
        collector.append(seq)
    return collector

def classify_collate(batch_samples: Tuple[List[List[int]], List[int], List[int]], tensor_type, pad: int) -> \
    Tuple[torch.tensor, torch.tensor]:
    """
    collate function used to pad a batch of sampled examples
    :param batch_samples: examples to pad
    :param tensor_type: return tensor type
    :param pad: pad index
    :return:
    """
    input_ids, sizes, labels = zip(*batch_samples)
    input_ids = classify_pad_bath(input_ids, sizes, pad)
    return tensor_type(input_ids), tensor_type(labels)


class ClassifyDataset(OmniDataset):
    def __init__(self, group: Dict, pad: int, shuffle: bool = True):
        super(ClassifyDataset, self).__init__(pad, shuffle)

        self.input_ids = group.get("ids")
        self.labels = group.get("labels")
        self.sizes = np.array(group.get("sizes"))
        self.epoch = 1

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, item) -> Tuple[List[int], int, int]:
        ids = self.input_ids[item]
        size = self.sizes[item]
        label = self.labels[item]

        return (ids, size, label)

    def num_tokens(self, index: int):
        return self.sizes[index]

    def size(self, index: int):
        return self.sizes[index]

    def collate(self, samples: Tuple[List[List[int]], List[int], List[int]]):
        return classify_collate(samples, self.tensor_type, self.pad)

    def ordered_indices(self) -> List[int]:
        if self.shuffle:
            indices = np.random.permutation(len(self))
        else:
            indices = np.arrange(len(self))
        return indices[np.argsort(self.sizes[indices], kind='mergesort')]

    def prefetch(self, indices: List[int]):
        self.input_ids.prefetch(indices)
        self.labels.prefetch(indices)
        self.sizes.prefetch(indices)

    def set_epoch(self, epoch):
        self.epoch = epoch


def get_classify_dataset(path_, group_names, max_tokens_per_batch: int,
                         pad: int,
                         shuffle: bool = True,
                         seed: int = 0,
                         max_sentence_length: int = 256,
                         max_sentences_per_batch: int = 50,
                         num_gpus: int = 1,
                         max_iter_length: int = 0,
                         num_workers: int = 0,
                         num_shards: int = 1,
                         shard_id: int = 0
                         ):

    dataset = load_obj_from_file(path_)[group_names]
    dataset = ClassifyDataset(dataset, pad, shuffle)

    with data_utils.numpy_seed(seed):
        indices = dataset.ordered_indices()

    # filter sentences too long
    indices = data_utils.filter_by_size(indices, dataset, min((max_sentence_length, max_tokens_per_batch)))

    # create mini-batch with given size constraint
    batch_sampler = data_utils.batch_by_size(
        indices,
        dataset.num_tokens,
        max_tokens=max_tokens_per_batch,
        max_sentences=max_sentences_per_batch,
        required_batch_size_multiple=num_gpus
    )

    # return a reusable iterator
    epoch_iter = iterators.EpochBatchIterator(
        dataset=dataset,
        collate_fn=dataset.collate,
        batch_sampler=batch_sampler,
        seed=seed,
        num_shards=num_shards,
        shard_id=shard_id,
        num_workers=num_workers,
        epoch=1,
        buffer_size=200,
        max_iter_len=max_iter_length
    )

    return epoch_iter


if __name__ == '__main__':
    print("implement test if needed")