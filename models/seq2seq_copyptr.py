from json import decoder, encoder
from typing import Optional
import torch
from torch.nn import Tensor
from copy_generator import CopyGenerator
from transformers import BartModel
from ..constants import TensorInputs


class Seq2SeqCopyPointer(torch.nn.Module):
    def __init__(self, pretrained: str, max_seq_len: int, bos_token_id: int, eos_token_id: int, pad_token_id):
        bart_model = BartModel.from_pretrained(pretrained)

        self.max_seq_len: int = max_seq_len
        self.eos_token_id: int = eos_token_id
        self.pad_token_id: int = pad_token_id
        self.bos_token_id: int = bos_token_id

        self.max_seq_len: int = bart_model.max_seq_len

        self.encoder = bart_model.encoder
        self.decoder = bart_model.decoder

        self.copy_generator = CopyGenerator()

    def forward(self, inputs: TensorInputs, targets: TensorInputs):
        """Supports token_ids only."""
        # Encode inputs
        encoder_hidden_states = self.encoder(input_ids=inputs.input_ids,
                                             attention_mask=inputs.attn_mask).last_hidden_state

        outputs = []
        # Decode
        for step in range(1, self.max_seq_len - 1):
            decoder_hidden_states = self.decoder(input_ids=targets.input_ids[:, :step],
                                                 atteention_mask=targets.attn_mask[:, :step],
                                                 encoder_hidden_states=encoder_hidden_states,
                                                 encoder_attention_mask=inputs.attn_mask).last_hidden_state
            decoder_hidden_states = decoder_hidden_states[:, -1:] # Get hidden_states of the last token

            # Copy or generate
            copy_probs = self.copy_generator(encoder_hidden_states, decoder_hidden_states)

            outputs.append(copy_probs)
        return torch.stach(outputs, dim=1)

    def predict(self, inputs: TensorInputs) -> Tensor:
        # Encode
        encoder_hidden_states = self.encoder(input_ids=inputs.input_ids,
                                             attention_mask=inputs.attn_mask).last_hidden_state

        # Init outputs with <BOS>
        outputs: Tensor = torch.tile(self.pad_token_id, (len(inputs), self.max_seq_len))
        outputs[:, 0] = self.bos_token_id

        # Decode
        for _ in range(self.max_max_seq_len):
            # Get non-terminal outputs
            indices = (outputs[:, -1] != self.eos_token_id).int() * (outputs[:, -1] != self.pad_token_id).int()
            indices = torch.nonzero(indices).reshape(-1)    # shape = [-1] or empty

            # Stop if meeting EOS or PAD
            if len(indices) == 0:
                break
            past_outputs = torch.index_select(outputs, 0, indices)

            decoder_hidden_states = self.decoder(
                input_ids=past_outputs,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=inputs.attn_mask.index_select(0, indices)).last_hidden_state
            decoder_hidden_states = decoder_hidden_states[:, -1:]   # Get hidden_states of the last token

            # Copy or generate
            # [batch_size, ontology_vocab_size + max_seq_len]
            outputs: Tensor = self.copy_generator(encoder_hidden_states.index_select(0, indices), decoder_hidden_states)

            # Update outputs
            outputs[indices] = torch.cat([past_outputs, outputs])

        return outputs
