from collections import OrderedDict

JIGSAW_MULTILINGUAL_LABELS_TO_IDX = OrderedDict([
    ("toxicity", 0),
    ("severe_toxicity", 1),
    ("obscene", 2),
    ("identity_attack", 3),
    ("insult", 4),
    ("threat", 5),
     ("sexual_explicit", 6)
])

JIGSAW_MULTILINGUAL_IDX_TO_LABEL = OrderedDict([(v, k) for k, v in JIGSAW_MULTILINGUAL_LABELS_TO_IDX.items()])


__all__ = [
    "JIGSAW_MULTILINGUAL_LABELS_TO_IDX",
    "JIGSAW_MULTILINGUAL_IDX_TO_LABEL"
]
