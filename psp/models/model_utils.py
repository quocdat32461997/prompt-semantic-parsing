import torch
from torch import Tensor
from typing import List
from psp.constants import IGNORED_INDEX


class FFN(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim_list: List[int], dropout: float):
        super().__init__()

        self.layers = torch.nn.ModuleList()
        for index, hidden_dim in enumerate(hidden_dim_list):
            self.layers.append(torch.nn.Linear(input_dim, hidden_dim))
            if index < len(hidden_dim_list) - 1:
                self.layers.append(torch.nn.ReLU())
                self.layers.append(torch.nn.Dropout(dropout))
            input_dim = hidden_dim

    def forward(self, hidden_states: Tensor):
        for layer in self.layers:
            hidden_states = layer(hidden_states)

        return hidden_states


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
