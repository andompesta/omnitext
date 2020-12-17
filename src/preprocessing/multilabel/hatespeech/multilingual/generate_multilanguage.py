from transformers import XLMRobertaTokenizer
from typing import Dict
import os
import pandas as pd
import argparse
from dynaconf import settings
from src.utils import save_obj_to_file

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_name", default="hatespeech")
    parser.add_argument("--db_version", default="0.1")
    parser.add_argument("--sentence_min_len", default=5)
    return parser.parse_args()


def encode(
        db_path: str,
        db_name: str,
        tokenizer: object,
) -> Dict[str, list]:
    ids = []
    labels = []
    sizes = []

    df = pd.read_csv(
        os.path.join(
            db_path,
            f"{db_name}.tsv"
        ),
        sep="\t",
    )

    for example in df.itertuples(index=False, name="exp"):
        id_ = tokenizer.encode(
            example.comment_text,
            truncation=True,
            max_length=512
        )
        ids.append(id_)
        labels.append((
            example.toxicity,
            example.severe_toxicity,
            example.obscene,
            example.identity_attack,
            example.insult,
            example.threat,
            example.sexual_explicit,
        ))
        sizes.append(len(id_))

    assert len(ids) == len(sizes) == len(labels)

    data = dict(
        ids=ids,
        sizes=sizes,
        labels=labels,
    )

    return data


if __name__ == '__main__':
    collector = {}

    collector["train"] = encode(
        db_path=os.path.join(
            settings.get("data_dir"),
            "hatespeech"
        ),
        db_name="train",
        tokenizer=XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
    )

    collector["eval"] = encode(
        db_path=os.path.join(
            settings.get("data_dir"),
            "hatespeech"
        ),
        db_name="eval",
        tokenizer=XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
    )

    collector["test"] = encode(
        db_path=os.path.join(
            settings.get("data_dir"),
            "hatespeech",
        ),
        db_name="test",
        tokenizer=XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
    )

    save_obj_to_file(
        os.path.join(
            settings.get("data_dir"),
            "multilabel",
            "jigsaw_multilingual.pt"
        ),
        collector
    )
