import torch
from torch import Tensor
from typing import List

class FFN(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim_list: List[int], dropout: float):
        super().__init__()

        self.layers = torch.nn.ModuleList()
        for index, hidden_dim in enumerate(hidden_dim_list):
            self.layers.append(torch.nn.Linear(input_dim, hidden_dim))
            if index < len(hidden_dim_list) - 1:
                self.layers.append(torch.nn.LeakyReLU()) # ReLU not converging
                self.layers.append(torch.nn.Dropout(dropout))
            input_dim = hidden_dim

    def forward(self, hidden_states: Tensor):
        for layer in self.layers:
            hidden_states = layer(hidden_states)

        return hidden_states