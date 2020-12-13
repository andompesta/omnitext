from datasets import load_dataset, Dataset
from collections import OrderedDict
from transformers import RobertaTokenizerFast
from typing import Dict
import os
from dynaconf import settings

from src.utils import save_obj_to_file, ensure_dir

def encode(
        data: Dataset,
        tokenizer: RobertaTokenizerFast,
) -> Dict[str, list]:
    idxs = []
    ids = []
    labels = []
    sizes = []

    for idx, example in enumerate(data):
        id_ = tokenizer.encode(
            example.get("text"),
            truncation=True,
            max_length=512
        )
        ids.append(id_)
        labels.append(example.get("label"))
        sizes.append(len(id_))
        idxs.append(idxs)

    assert len(idxs) == len(ids) == len(sizes) == len(labels)

    data = dict(
        ids=ids,
        sizes=sizes,
        labels=labels,
        idxs=idxs
    )

    return data


if __name__ == '__main__':
    dataset = load_dataset("imdb")
    train = dataset.get("train").shuffle()
    eval = dataset.get("test")
    eval = eval.train_test_split(.5)
    test = eval.get("test")
    eval = eval.get("train")


    labels_to_idx = OrderedDict()
    for l, in zip(train.features['label'].names):
        labels_to_idx[l] = train.features['label'].str2int(l)
    idx_to_labels = OrderedDict([(v, k) for k,v in labels_to_idx.items()])

    print(labels_to_idx)
    print(idx_to_labels)

    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-import")

    collector = {}

    collector["train"] = encode(
        train,
        tokenizer
    )

    collector["eval"] = encode(
        eval,
        tokenizer
    )

    collector["test"] = encode(
        test,
        tokenizer
    )

    save_obj_to_file(
        ensure_dir(os.path.join(settings.get("data_dir"), "classification", "imdb.pt")),
        collector
    )