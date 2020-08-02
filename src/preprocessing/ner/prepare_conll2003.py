# code taken from https://github.com/ShannonAI/mrc-for-flat-nested-ner/blob/master/data_preprocess/generate_mrc_dataset.py

import argparse
import json
from typing import List, Tuple, Dict
from dynaconf import settings
from os import path

from src.preprocessing.ner import CONLL_LABEL_QUERY, CONLL_LABLES

def load_conll(path_: str) -> List[Tuple[List[str], List[str]]]:
    """
    Desc:
        load data in conll format
    Returns:
        [([word1, word2, word3, word4], [label1, label2, label3, label4]),
        ([word5, word6, word7, wordd8], [label5, label6, label7, label8])]
    """
    dataset = []
    with open(path_, "r") as f:
        words, tags = [], []
        # for each line of the file correspond to one word and tag
        for line in f:
            if line != "\n":
                # line = line.strip()
                word, _, _, tag = line.split(" ")
                word = word.strip()
                tag = tag.strip()
                try:
                    if len(word) > 0 and len(tag) > 0:
                        word, tag = str(word), str(tag)
                        words.append(word)
                        tags.append(tag)
                except Exception as e:
                    print("an exception was raise! skipping a word")
            else:
                if len(words) > 0:
                    assert len(words) == len(tags)
                    dataset.append((words, tags))
                    words, tags = [], []

    return dataset

def get_span_labels(sentence_tags, inv_label_mapping=None):
    """
    Desc:
        get from token_level labels to list of entities,
        it doesnot matter tagging scheme is BMES or BIO or BIOUS
    Returns:
        a list of entities
        [(start, end, labels), (start, end, labels)]
    """

    if inv_label_mapping:
        sentence_tags = [inv_label_mapping[i] for i in sentence_tags]

    span_labels = []
    last = "O"
    start = -1
    for i, tag in enumerate(sentence_tags):
        pos, _ = (None, "O") if tag == "O" else tag.split("-")
        if (pos == "S" or pos == "B" or tag == "O") and last != "O":
            span_labels.append((start, i - 1, last.split("-")[-1]))
        if pos == "B" or pos == "S" or last == "O":
            start = i
        last = tag

    if sentence_tags[-1] != "O":
        span_labels.append((start, len(sentence_tags) -1 , sentence_tags[-1].split("-")[-1]))

    return span_labels


def transform_examples_to_qa_features(query_map: Dict[str, str], entity_labels: Dict[str, int], data_instances: List,
                                      entity_sign="flat"):
    """
    convert_examples to qa features
    """

    mrc_ner_dataset = []

    if entity_sign.lower() == "flat":
        tmp_qas_id = 0
        for idx, (word_lst, label_lst) in enumerate(data_instances):
            candidate_span_label = get_span_labels(label_lst)
            tmp_query_id = 0
            for tmp_label, label_idx in entity_labels.items():
                tmp_query_id += 1
                tmp_query = query_map[tmp_label]
                tmp_context = " ".join(word_lst)

                tmp_start_pos = []
                tmp_end_pos = []
                tmp_entity_pos = []

                start_end_label = [(start, end) for start, end, label_content in candidate_span_label if
                                   label_content == tmp_label]

                if len(start_end_label) != 0:
                    for span_item in start_end_label:
                        start_idx, end_idx = span_item
                        tmp_start_pos.append(start_idx)
                        tmp_end_pos.append(end_idx)
                        tmp_entity_pos.append("{};{}".format(str(start_idx), str(end_idx)))
                    tmp_impossible = False
                else:
                    tmp_impossible = True

                mrc_ner_dataset.append({
                    "qas_id": "{}.{}".format(str(tmp_qas_id), str(tmp_query_id)),
                    "query": tmp_query,
                    "context": tmp_context,
                    "entity_label": tmp_label,
                    "start_position": tmp_start_pos,
                    "end_position": tmp_end_pos,
                    "span_position": tmp_entity_pos,
                    "impossible": tmp_impossible
                })
            tmp_qas_id += 1

    elif entity_sign.lower() == "nested":
        tmp_qas_id = 0
        for idx, data_item in enumerate(data_instances):
            tmp_query_id = 0
            for tmp_label, label_idx in entity_labels.items():
                tmp_query_id += 1
                tmp_query = query_map[tmp_label]
                tmp_context = data_item["context"]

                tmp_start_pos = []
                tmp_end_pos = []
                tmp_entity_pos = []

                start_end_label = data_item["label"][tmp_label] if tmp_label in data_item["label"].keys() else -1

                if start_end_label == -1:
                    tmp_impossible = True
                else:
                    for start_end_item in data_item["label"][tmp_label]:
                        start_end_item = start_end_item.replace(",", ";")
                        start_idx, end_idx = [int(ix) for ix in start_end_item.split(";")]
                        tmp_start_pos.append(start_idx)
                        tmp_end_pos.append(end_idx)
                        tmp_entity_pos.append(start_end_item)
                    tmp_impossible = False

                mrc_ner_dataset.append({
                    "qas_id": "{}.{}".format(str(tmp_qas_id), str(tmp_query_id)),
                    "query": tmp_query,
                    "context": tmp_context,
                    "entity_label": tmp_label,
                    "start_position": tmp_start_pos,
                    "end_position": tmp_end_pos,
                    "span_position": tmp_entity_pos,
                    "impossible": tmp_impossible
                })
            tmp_qas_id += 1
    else:
        raise ValueError("Please Notice that entity_sign can only be flat OR nested. ")

    return mrc_ner_dataset


def __parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="conll2003")
    parser.add_argument("--queries", type=Dict[str, str], default=CONLL_LABEL_QUERY)
    parser.add_argument("--labels", type=Dict[str, int], default=CONLL_LABLES)
    parser.add_argument("--type", type=str, default="flat")

    return parser.parse_args()

if __name__ == '__main__':
    args = __parse_args()
    for dset in ["train", "dev", "test"]:
        source_data = load_conll(path.join(settings.get("data_dir"), args.dataset_name, f"eng.{dset}"))
        target_data = transform_examples_to_qa_features(args.queries, args.labels, source_data, entity_sign=args.type)

        with open(path.join(settings.get("data_dir"), args.dataset_name, f"mrc-ner.{dset}"), "w") as f:
            json.dump(target_data, f, sort_keys=True, ensure_ascii=False, indent=2)