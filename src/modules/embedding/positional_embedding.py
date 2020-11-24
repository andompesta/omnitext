from .learned_positional_embedding import LearnedPositionalEmbedding
from .sinusoidal_positional_embedding import SinusoidalPositionalEmbedding
from .partially_fixed_embedding import PartiallyFixedLearnedPositionalEmbedding
from typing import Optional


def PositionalEmbedding(
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int,
        type: str,
        initializer_range: Optional[float] = 0.02,
        trainable_num_embeddings: int = 0,
        fixed_num_embeddings: int = 0
):
    if type == "learned":
        m = LearnedPositionalEmbedding(
            num_embeddings,
            embedding_dim,
            padding_idx,
            initializer_range
        )

    elif type == "partially_fixed":
        assert trainable_num_embeddings > 0
        assert num_embeddings - fixed_num_embeddings == trainable_num_embeddings

        m = PartiallyFixedLearnedPositionalEmbedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
            initializer_range=initializer_range,
            trainable_num_embeddings=trainable_num_embeddings,
            fixed_num_embeddings=fixed_num_embeddings
        )

    elif type == "sinusoidal":
        m = SinusoidalPositionalEmbedding(
            num_embeddings,
            embedding_dim,
            padding_idx
        )
    else:
        raise NotImplementedError(f"embedding type {type} not yet implemented")

    return m
