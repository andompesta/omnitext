from collections.abc import Iterable
import contextlib
import numpy as np


def batchfy(iterable, n=1):
    l = len(iterable)
    for idx in range(0, l, n):
        yield iterable[idx: min(idx + n, l)]

def get_singel_batch(dl):
    ret = []
    for batch in dl:
        ret.append(batch)
        break
    return ret



@contextlib.contextmanager
def numpy_seed(seed, *addl_seeds):
    """
    Context manager which seeds the NumPy PRNG with the specified seed and restores the state afterward
    :param seed: seed value
    :param addl_seeds:
    :return:
    """
    if seed is None:
        yield
        return
    if len(addl_seeds) > 0:
        seed = int(hash((seed, *addl_seeds)) % 1e6)
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)

def collect_filtered(function, iterable, filtered: list):
    """
    Similar to :func: `filter` but collectes filtered elements in ``filtered``.
    :param function: evaluation function
    :param iterable: iterable to filter
    :param filtered: return collection
    :return:
    """
    for el in iterable:
        if function(el):
            yield el
        else:
            filtered.append(el)

def _filter_by_size_dynamic(indices, size_fn, max_positions):
    def compare_leq(a, b):
        return a <= b if not isinstance(a, tuple) else max(a) <= b

    def check_size(idx):
        if isinstance(max_positions, float) or isinstance(max_positions, int):
            return size_fn(idx) <= max_positions
        elif isinstance(max_positions, dict):
            idx_size = size_fn(idx)
            assert isinstance(idx_size, dict)
            intersect_keys = set(max_positions.keys()) & set(idx_size.keys())
            return all(
                all(a is None or b is None or a <= b
                    for a, b in zip(idx_size[key], max_positions[key]))
                for key in intersect_keys
            )
        else:
            # Hacky as heck, for the specific case of multilingual training with RoundRobin.
            if isinstance(size_fn(idx), dict) and isinstance(max_positions, tuple):
                return all(
                    a is None or b is None or compare_leq(a, b)
                    for a, b in zip(size_fn(idx).values(), max_positions)
                )
            # For MultiCorpusSampledDataset, will generalize it later
            if not isinstance(size_fn(idx), Iterable):
                return all(size_fn(idx) <= b for b in max_positions)
            return all(
                a is None or b is None or a <= b
                for a, b in zip(size_fn(idx), max_positions)
            )
    ignored = []
    itr = collect_filtered(check_size, indices, ignored)
    indices = np.fromiter(itr, dtype=np.int64, count=-1)
    return indices, ignored


def filter_by_size(indices, dataset, max_positions: int, raise_exception=False):
    """
    Filter indices based on their size.
    Args:
        indices (List[int]): ordered list of dataset indices
        dataset (OmniDataset): omnitext dataset instance
        max_positions (tuple): filter elements larger than this size.
            Comparisons are done component-wise.
        raise_exception (bool, optional): if ``True``, raise an exception if
            any elements are filtered (default: False).
    """
    if isinstance(max_positions, float) or isinstance(max_positions, int):
        if hasattr(dataset, 'sizes') and isinstance(dataset.sizes, np.ndarray):
            ignored = indices[dataset.sizes[indices] > max_positions].tolist()
            indices = indices[dataset.sizes[indices] <= max_positions]
        elif hasattr(dataset, 'sizes') and isinstance(dataset.sizes, list) and len(dataset.sizes) == 1:
            ignored = indices[dataset.sizes[0][indices] > max_positions].tolist()
            indices = indices[dataset.sizes[0][indices] <= max_positions]
        else:
            indices, ignored = _filter_by_size_dynamic(indices, dataset.size, max_positions)
    else:
        indices, ignored = _filter_by_size_dynamic(indices, dataset.size, max_positions)

    if len(ignored) > 0 and raise_exception:
        raise Exception((
            'Size of sample #{} is invalid (={}) since max_positions={}, '
            'skip this example with --skip-invalid-size-inputs-valid-test'
        ).format(ignored[0], dataset.size(ignored[0]), max_positions))
    if len(ignored) > 0:
        print('WARNING: {} samples have invalid sizes and will be skipped, '
              'max_positions={}, first few sample ids={}'.format(len(ignored), max_positions, ignored[:10]))
    return indices


def batch_by_size(
    indices, num_tokens_fn, max_tokens=None, max_sentences=None,
    required_batch_size_multiple=1, fixed_shapes=None,
):
    """
    Yield mini-batches of indices bucketed by size. Batches may contain
    sequences of different lengths.
    Args:
        indices (List[int]): ordered list of dataset indices
        num_tokens_fn (callable): function that returns the number of tokens at
            a given index
        max_tokens (int, optional): max number of tokens in each batch
            (default: None).
        max_sentences (int, optional): max number of sentences in each
            batch (default: None).
        required_batch_size_multiple (int, optional): require batch size to
            be less than N or a multiple of N (default: 1).
        fixed_shapes (List[Tuple[int, int]], optional): if given, batches will
            only be created with the given shapes. *max_sentences* and
            *required_batch_size_multiple* will be ignored (default: None).
    """
    try:
        from .data_utils_fast import (
            batch_by_size_fast, batch_fixed_shapes_fast,
        )
    except ImportError:
        raise ImportError(
            'Please build Cython components with: `pip install --editable .` '
            'or `python setup.py build_ext --inplace`'
        )

    max_tokens = max_tokens if max_tokens is not None else -1
    max_sentences = max_sentences if max_sentences is not None else -1
    bsz_mult = required_batch_size_multiple

    if not isinstance(indices, np.ndarray):
        indices = np.fromiter(indices, dtype=np.int64, count=-1)

    if fixed_shapes is None:
        return batch_by_size_fast(
            indices, num_tokens_fn, max_tokens, max_sentences, bsz_mult,
        )
    else:
        fixed_shapes = np.array(fixed_shapes, dtype=np.int64)
        sort_order = np.lexsort([
            fixed_shapes[:, 1].argsort(),  # length
            fixed_shapes[:, 0].argsort(),  # bsz
        ])
        fixed_shapes_sorted = fixed_shapes[sort_order]
        return batch_fixed_shapes_fast(indices, num_tokens_fn, fixed_shapes_sorted)


def process_bpe_symbol(sentence: str, bpe_symbol: str):
    if bpe_symbol == 'sentencepiece':
        sentence = sentence.replace(' ', '').replace('\u2581', ' ').strip()
    elif bpe_symbol == '_EOW':
        sentence = sentence.replace(' ', '').replace('_EOW', ' ').strip()
    elif bpe_symbol is not None:
        sentence = (sentence + ' ').replace(bpe_symbol, '').rstrip()
    return sentence