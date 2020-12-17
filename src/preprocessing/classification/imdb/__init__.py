from collections import OrderedDict

IMDB_LABEL_TO_IDX = OrderedDict([('neg', 0), ('pos', 1)])
IMDB_IDX_TO_LABEL = OrderedDict([(0, 'neg'), (1, 'pos')])

__all__ = [
    "IMDB_LABEL_TO_IDX",
    "IMDB_IDX_TO_LABEL"
]
