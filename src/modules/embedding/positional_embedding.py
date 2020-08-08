from .learned_positional_embedding import LearnedPositionalEmbedding
from .sinusoidal_positional_embedding import SinusoidalPositionalEmbedding

from typing import Optional
def PositionalEmbedding(
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int,
        learned: bool = False,
        initializer_range: Optional[float] = 0.02
):
    # if padding_idx is specified then offset the embedding ids by
    # this index and adjust num_embeddings appropriately
    if padding_idx is not None:
        num_embeddings = num_embeddings + padding_idx + 1

    if learned:
        m = LearnedPositionalEmbedding(num_embeddings,
                                       embedding_dim,
                                       padding_idx,
                                       initializer_range)
    else:
        m = SinusoidalPositionalEmbedding(num_embeddings,
                                       embedding_dim,
                                       padding_idx)
    return m