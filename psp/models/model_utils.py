import torch
from torch import Tensor
from typing import List
from psp.constants import IGNORED_INDEX

def pad_tensors(tensors: Tensor, batch_size: int, max_length: int, value: int):
    """
    Pad 2D tensors
    """
    assert len(tensors.shape) == 2
    return torch.cat(
        [
            tensors,
            torch.full((batch_size, max_length - tensors.shape[-1]), value, device=tensors.device),
        ],
        dim=-1,
    )

def ignore_tokens(tensors: Tensor, tbi_tokens: List[int]) -> Tensor:
    """Mask-select tokens not in tbi-tokens"""
    """
    Args:
        tensors: Tensor to be mask-selected
        tbi_tokens: List of to-be-ignored tokens
    """
    # Cast to tensor
    tbi_tokens = torch.tensor(tbi_tokens)

    # Get mask of tokens to be ignored
    mask = torch.isin(tensors, tbi_tokens, invert=True)

    return torch.where(mask, tensors, torch.full(tensors.shape, IGNORED_INDEX))

def get_arange_matrix(shape: torch.Size) -> Tensor:
    """Create a 2D arange matrix
        e.g. 
            [[1,2,3,4]
             [1,2,3,4]]
    """
    assert shape[0] > 0 and shape[1] > 0
    return torch.tile(torch.arange(shape[1], [shape[0], 1]))