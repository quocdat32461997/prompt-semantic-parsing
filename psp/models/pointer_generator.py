from typing import List
import torch
import torch.nn.functional as F
from torch import Tensor

from psp.constants import RunMode


class PointerGenerator(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        ontology_vocab_ids: List[int],
        input_dim: int,
        hidden_dim_list: List[int],
        num_heads: int = 12,
        dropout: float = 0.3,
    ) -> None:
        """
        Args:
            vocab_size: number of original vocabs + ontology vocabs
            ontology_vocab_ids: token-ids of ontology vocabs
            input_dim: embedding dim
            hidden_dims_list: dim-list of hidden layers (including output dimenstion but not embedding dim).
                The output dimension should the number of ontology tokens.
            num_heads: number of attention heads. Default to to 12 as in BART
            dropout: float. Default to 0.3 as in Low-Resource Domain Adaptation for Compositional Task-Oriented Semantic Parsing
        """
        super().__init__()
        self.vocab_size: int = vocab_size
        self.ontology_vocab_ids: Tensor = torch.tensor(
            ontology_vocab_ids
        )  # [ontology_vocab]

        self.copier = torch.nn.ModuleList(
            [
                torch.nn.MultiheadAttention(
                    num_heads=num_heads,
                    embed_dim=input_dim,
                    dropout=dropout,
                    batch_first=True,
                ),
                torch.nn.Linear(input_dim, 1),
            ]
        )

        self.generator = torch.nn.ModuleList()
        for index, hidden_dim in enumerate(hidden_dim_list):
            self.generator.append(torch.nn.Linear(input_dim, hidden_dim))
            if index < len(hidden_dim_list) - 1:
                self.generator.append(torch.nn.ReLU())
                self.generator.append(torch.nn.Dropout(dropout))
            input_dim = hidden_dim

        self.copy_head = torch.nn.Linear(input_dim, 1)

    def forward(
        self,
        source_input_ids: Tensor,
        encoder_hidden_states: Tensor,
        encoder_attn_mask: Tensor,
        decoder_hidden_states: Tensor,
        run_mode: RunMode,
    ) -> Tensor:
        """
        Args:
            source_input_ids: Token-ids from the source sequence [batch_size, source_seq_len]
            encoder_hidden_states: [batch_size, source_seq_len, embed_size]
            encoder_attn_mask: [batch_size, source_seq_len]
            decoder_hidden_states: [batch_size, max_seq_len or 1, embed_size]
            run_mode: RunMode to switch between log_softmax for training of softmax for inference.
        """

        # Generate ontology
        # [batch_size, max_seq_len, ontology_vocab_size]
        ontology_logits: Tensor = self.generator(decoder_hidden_states)

        # Copy from source sequences
        # context_outputs: [batch_size, max_seq_len or 1, embed_dim]
        # copy_source_probs: [batch_ssize, max_seq_len or 1, source_seq_len]
        context_outputs, copy_source_probs = self.copier(
            query=encoder_hidden_states,
            key=decoder_hidden_states,
            attention_mask=encoder_attn_mask,
            need_weights=True,
        )
        copy_logits: Tensor = self.copy_head(
            context_outputs
        )  # [batch_size, max_seq_len or 1, 1]

        # Final predictions if in either eval or test modes
        softmax_func = F.log_softmax if run_mode == RunMode.TRAIN else F.softmax
        ontology_probs: Tensor = softmax_func(
            ontology_logits
        )  # [batch_size, max_seq_len or 1, ontology_vocab_size]
        copy_probs: Tensor = softmax_func(
            copy_logits
        )  # [batch_size, max_seq_len or 1, 1]

        # Get final probs
        copy_source_probs *= (
            copy_probs  # [batch_size, max_seq_len or 1, source_seq_len]
        )
        ontology_probs *= (
            1 - copy_probs
        )  # [batch_size, max_seq_len or 1, ontology_vocab_size]

        # Parse into full vocabs
        # [batch_size, max_seq_lne or 1, vocab_size]
        vocab_probs: Tensor = torch.zero(
            copy_probs.shape[:2] + [self.vocab_size], dtype=copy_probs.dtype
        )

        source_input_ids = source_input_ids.unsqueeze(
            1
        )  # [batch_size, 1, source_seq_len]
        # [batch_size, max_seq_len or 1, source_seq_len]
        source_input_ids = torch.tile(source_input_ids, [1, copy_probs.shape[1], 1])
        vocab_probs = vocab_probs.scatter_add_(
            dim=-1, index=source_input_ids, src=copy_source_probs
        )

        ontology_vocab_ids: Tensor = self.ontology_vocab_ids.reshape(
            [1, 1, -1]
        )  # [1, 1, ontology_vocab]
        # [batch_size, max_seq_len or 1, ontology_vocab]
        ontology_vocab_ids = torch.tile(ontology_vocab_ids, vocab_probs.shape[:2] + [1])
        vocab_probs = vocab_probs.scatter_add_(
            dim=-1, index=self.ontology_vocab_ids, src=ontology_probs
        )

        return vocab_probs
