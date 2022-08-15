from typing import List
import torch
import torch.nn.functional as F
from torch import Tensor


class CopyGenerator(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim_list: List[int], num_heads: int = 12, dropout: float = 0.3) -> None:
        """
        Args:
            input_dim: embedding dim
            hidden_dims: dim-list of hidden layers (including output dimenstion but not embedding dim)
            num_heads: number of attention heads. Default to to 12 as in BART
            dropout: float. Default to 0.3 as in Low-Resource Domain Adaptation for Compositional Task-Oriented Semantic Parsing
        """
        super().__init__()
        self.copier = torch.nn.ModuleList([
            torch.nn.MultiheadAttention(num_heads=num_heads,
                                        embed_dim=input_dim,
                                        dropout=dropout,
                                        batch_first=True),
            torch.nn.Linear(input_dim, 1)])

        self.generator = torch.nn.ModuleList()
        for index, hidden_dim in enumerate(hidden_dim_list):
            self.generator.append(torch.nn.Linear(input_dim, hidden_dim))
            if index < len(hidden_dim_list) - 1:
                self.generator.append(torch.nn.ReLU())
                self.generator.append(torch.nn.Dropout(dropout))
            input_dim = hidden_dim

    def forward(self, encoder_outputs: Tensor, decoder_outputs: Tensor) -> Tensor:
        """
        Args:
            encoder_outputs: [batch_size, source_seq_len, embed_size]
            decoder_outputs: [batch_size, 1, embed_size]

        Returns:
            [batch_size, ontology_vocab_size + source_seq_len]
        """

        # Get probs to generate ontology
        ontology_probs = self.generator(decoder_outputs)
        ontology_probs = F.softmax(ontology_probs)

        # Get attention weights and context over source sequence
        from_source_probs, context_outputs = self.copier(encoder_outputs, decoder_outputs)

        # Get copy probs
        copy_probs = F.sigmoid(context_outputs)

        # Get final probs
        from_source_probs *= copy_probs
        ontology_probs *= (1 - copy_probs)

        return torch.stack([ontology_probs, from_source_probs], dim=-1)
