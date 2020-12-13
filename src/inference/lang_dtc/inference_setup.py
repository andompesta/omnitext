import numpy as np
from typing import List, Tuple
from functools import partial
import fasttext
from src.preprocessing.clean_data import remove_double_spaces, remove_non_utf8

def __preprocess__(text: str) -> str:
    text = text.lower()
    text = text.replace("\n", ". ")
    text = remove_non_utf8(text)
    text = remove_double_spaces(text)
    return text

def preprocess(texts: List[str]) -> List[str]:
    return [__preprocess__(t) for t in texts]


def __postprocess__(p: List[str], c: np.array) -> Tuple[str, float]:
    return p[0].replace("__label__", ""), float(c[0])

def postprocess(pred_t: List[List[str]], conf_t: List[np.array]) -> Tuple[List[str], List[float]]:
    assert len(pred_t) == len(conf_t)
    pred_conf_t = [__postprocess__(p, c) for p, c in zip(pred_t, conf_t)]
    pred_t, conf_t = list(zip(*pred_conf_t))
    return list(pred_t), list(conf_t)


def inference(texts: List[str], model) -> Tuple[List[List[str]], List[np.array]]:
    pred_t, conf_t = model.predict(texts)
    return pred_t, conf_t

def inference_setup(
        model_path: str,
        **kwargs
):
    model = LangDtcModel(model_path)
    inference_fn = partial(inference, model=model)
    return preprocess, inference_fn, postprocess



class LangDtcModel(object):
    """
    FastText wrapper to allow parallel computations
    """
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = fasttext.load_model(model_path)

    def predict(self, texts):
        return self.model.predict(texts)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["model"] = None
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self.model = fasttext.load_model(self.model_path)