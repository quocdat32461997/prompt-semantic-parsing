from typing import List
import torch
import torch.nn.functional as F
from torch import Tensor

from psp.constants import ParseOutputs


class CopyGenerator(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim_list: List[int], num_heads: int = 12, dropout: float = 0.3) -> None:
        """
        Args:
            input_dim: embedding dim
            hidden_dims_list: dim-list of hidden layers (including output dimenstion but not embedding dim).
                The output dimension should the number of ontology tokens.
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

        self.copy_head = torch.nn.Linear(input_dim, 1)

    def forward(self, encoder_hidden_states: Tensor, encoder_attn_mask: Tensor, decoder_hidden_states: Tensor) -> ParseOutputs:
        """
        Args:
            encoder_hidden_states: [batch_size, source_seq_len, embed_size]
            decoder_hidden_states: [batch_size, max_seq_len or 1, embed_size]
        """

        # Generate ontology
        # [batch_size, max_seq_len, ontology_vocab_size]
        ontology_logits: Tensor = self.generator(decoder_hidden_states)

        # Copy from source sequences
        # context_outputs: [batch_size, max_seq_len or 1, embed_dim]
        # copy_source_probs: [batch_ssize, max_seq_len or 1, source_seq_len]
        context_outputs, copy_source_probs = self.copier(query=encoder_hidden_states, key=decoder_hidden_states,
                                                         attention_mask=encoder_attn_mask, need_weights=True)
        copy_logits: Tensor = self.copy_head(context_outputs)   # [batch_size, max_seq_len or 1, 1]

        # Final predictions if in either eval or test modes
        ontology_probs: Tensor = F.softmax(ontology_logits)  # [batch_size, max_seq_len or 1, ontology_vocab_size]
        copy_probs: Tensor = F.sigmoid(copy_logits)  # [batch_size, max_seq_len or 1, 1]

        # Get final probs
        copy_source_probs *= copy_probs
        ontology_probs *= (1 - copy_probs)

        return ParseOutputs(ontology_probs, copy_source_probs, copy_probs)

    def generate_parse(self, encoder_hidden_states: Tensor, encoder_attn_mask: Tensor, decoder_hidden_states: Tensor, source_input_ids: Tensor) -> Tensor:
        """
        Args:
            encoder_hidden_states: [batch_size, source_seq_len, embed_size]
            encoder_attn_mask: [batch_size, source_seq_len]
            decoder_hidden_states: [batch_size, 1, embed_size]
            source_input_ids: [batch_size, source_seq_len]
        """

        assert decoder_hidden_states.shape[1] == 1

        batch_size: int = len(source_input_ids)

        # Generating parse
        parse_outputs: ParseOutputs = self.forward(encoder_hidden_states, encoder_attn_mask, decoder_hidden_states)

        # Get to-copy source tokens or to-generate ontology tokens
        # _tokens: [batch_size, 1]; _indices: [batch_size, 1]; _probs: [batch_size, 1]
        source_probs, source_indices = torch.max(parse_outputs.copy_source_probs, dim=-1)
        source_tokens: Tensor = source_input_ids[list(torch.arange(
            batch_size)), source_indices.reshape([-1])]

        ontology_probs, ontology_tokens = torch.max(parse_outputs.ontology_probs, dim=-1)

        return torch.where(source_probs > ontology_probs, source_tokens, ontology_tokens)   # [batch_size, 1]
