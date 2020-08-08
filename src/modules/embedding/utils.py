import torch
from typing import Optional

def create_position_ids_from_input_ids(
        input_ids: torch.Tensor,
        padding_idx: Optional[int]
) -> torch.Tensor:
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at
    padding_idx+1. Padding symbols are ignored. This is modified from fairseq's

    :param input_ids: tokens ids
    :param padding_idx: Pad index
    :return: positional embedding ids
    """
    # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = torch.cumsum(mask, dim=1).type_as(mask) * mask
    if padding_idx is not None:
        return incremental_indices.long() + padding_idx
    else:
        return incremental_indices.long()