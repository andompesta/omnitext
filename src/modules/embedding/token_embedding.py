from torch import nn, Tensor
import torch.nn.functional as F

from typing import Optional

class TokenEmbedding(nn.Embedding):
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int, initializer_range: float,
                 scale: Optional[float] = None):
        super(TokenEmbedding, self).__init__(num_embeddings, embedding_dim, padding_idx)
        self.initializer_range = initializer_range
        self.scale = scale

    def forward(self, input_ids: Tensor) -> Tensor:
        """Input is expected to be of size [bsz x seqlen]."""
        emb = F.embedding(
            input_ids,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )

        if self.scale is not None:
            emb *= self.scale

        return emb