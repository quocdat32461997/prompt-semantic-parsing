from typing import List
import torch
import torch.nn.functional as F
from torch import Tensor

from psp.constants import RunMode
from psp.models.model_utils import FFN

class Generator(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim_list: List[int], dropout: float):
        super(Generator, self).__init__()

        self.ffn = FFN(input_dim=input_dim, hidden_dim_list=hidden_dim_list, dropout=dropout)

    def forward(self, inputs):
        """
        Returns:
            - outputs: probs of shape [batch_size, max_seq_len, ontology_vocab_size]
        """
        outputs = self.ffn(inputs)
        outputs = F.softmax(outputs, dim=-1)

        return outputs

class CopyHead(torch.nn.Module):
    def __init__(self, input_dim: int):
        super(CopyHead, self).__init__()
        self.linear = torch.nn.Linear(input_dim, 1)

    def forward(self, inputs):
        """
        Args:
            inputs: context ouputs from multi-head-attention module. 
                Shape [batch_size, max_seq_len, embed_dim]
        Returns:
            outputs: scalar probs to copy from source inputs.
                Shape [batch_size, max_seq_len, 1]
        """
        outputs = self.linear(inputs)
        outputs = F.sigmoid(outputs)

        return outputs

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
            vocab_size: number of regular words + ontology vocabs
            ontology_vocab_ids: token-ids of ontology vocabs
            input_dim: embedding dim
            hidden_dims_list: dim-list of hidden layers (including output dimenstion but not embedding dim).
                The output dimension should the number of ontology tokens.
            num_heads: number of attention heads. Default to to 12 as in BART
            dropout: float. Default to 0.3 as in Low-Resource Domain Adaptation for Compositional Task-Oriented Semantic Parsing
        """
        super(PointerGenerator, self).__init__()
        self.vocab_size: int = vocab_size

        self.ontology_vocab_ids: Tensor = torch.tensor(
            ontology_vocab_ids
        ).reshape([1, 1, -1])  # [1, 1, ontology_vocab]
        
        # copier
        self.mha = torch.nn.MultiheadAttention(
            num_heads=num_heads,
            embed_dim=input_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.copy_head = CopyHead(input_dim=input_dim)

        # genrator
        self.generator = Generator(
            input_dim=input_dim, hidden_dim_list=hidden_dim_list, dropout=dropout
        )

    def forward(
        self,
        source_input_ids: Tensor,
        encoder_hidden_states: Tensor,
        encoder_attn_mask: Tensor,
        decoder_hidden_states: Tensor,
        run_mode: RunMode = RunMode.TRAIN,
        device: str = None,
    ) -> Tensor:
        """
        Args:
            source_input_ids: Token-ids from the source sequence [batch_size, source_seq_len]
            encoder_hidden_states: [batch_size, source_seq_len, embed_size]
            decoder_hidden_states: [batch_size, max_seq_len or 1, embed_size]
        Returns:
            vocab_probs: Tensor of shape [batch_size, max_seq_len or 1, ontology_vocab_size + source_seq_len]
        """

        if not device:
            device = source_input_ids.device
        
        # Generate probs to generate ontology tokens
        # [batch_size, max_seq_len or 1, ontology_vocab_size]
        ontology_probs: Tensor = self.generator(decoder_hidden_states)

        # Copy from source sequences
        # context_outputs: [batch_size, max_seq_len or 1, embed_dim]
        # copy_source_probs: [batch_size, max_seq_len or 1, source_seq_len]
        context_outputs, from_source_probs = self.mha(
            query=decoder_hidden_states,
            key=encoder_hidden_states,
            key_padding_mask=encoder_attn_mask,
            value=encoder_hidden_states,
            need_weights=True,
        )

        # Get probs to copy from source
        # copy_probs: [batch_size, max_seq_len or 1, 1]
        copy_probs = self.copy_head(context_outputs)

        # Get final probs
        from_source_probs = (
            from_source_probs
            * copy_probs  # [batch_size, max_seq_len or 1, source_seq_len]
        )
        ontology_probs = ontology_probs * (
            1 - copy_probs
        )  # [batch_size, max_seq_len or 1, ontology_vocab_size]

        # Parse into full vocabs
        # [batch_size, max_seq_len or 1, vocab_size]
        vocab_probs: Tensor = torch.zeros(
            list(from_source_probs.shape[:2]) + [self.vocab_size],
            dtype=copy_probs.dtype
        ).to(device)

        # Get indices to fill token-logits into vocab_probs
        # [batch_size, max_seq_len, source_seq_len]
        source_input_ids = source_input_ids.unsqueeze(1)
        source_input_ids = torch.tile(source_input_ids, [1, copy_probs.shape[1], 1])
        
        # [batch_size, max_seq_len or 1, ontology_vocab]
        ontology_vocab_ids: Tensor = torch.tile(
            self.ontology_vocab_ids, list(vocab_probs.shape[:2]) + [1]
        ).to(device)

        if run_mode == RunMode.EVAL:
            # [batch_size, max_seq_len, ontology_vocab_size + source_seq_len]
            probs: Tensor = torch.concat([ontology_probs, from_source_probs], dim=-1)

            # Apply softmax
            probs = F.softmax(probs, dim=-1)

            # Partition back to source-tokens and ontology-tokens
            ontology_probs = probs[..., :ontology_vocab_ids.shape[-1]]
            from_source_probs = probs[..., ontology_vocab_ids.shape[-1]:]

        vocab_probs = vocab_probs.scatter_add(
            dim=-1, index=source_input_ids, src=from_source_probs
        ) 
        vocab_probs = vocab_probs.scatter_add_(
            dim=-1, index=ontology_vocab_ids, src=ontology_probs
        )
    
        return vocab_probs
