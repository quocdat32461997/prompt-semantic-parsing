import torch
from torch import Tensor
from transformers import BartModel

from psp.models.copy_generator import CopyGenerator
from psp.constants import TensorInputs


class Seq2SeqCopyPointer(torch.nn.Module):
    def __init__(self, pretrained: str, ontology_vocab_size: int,
            bos_token_id: int, eos_token_id: int, pad_token_id):
        super().__init__()
        bart_model = BartModel.from_pretrained(pretrained)

        self.eos_token_id: int = eos_token_id
        self.pad_token_id: int = pad_token_id
        self.bos_token_id: int = bos_token_id

        self.max_seq_len: int = bart_model.config.max_position_embeddings

        self.encoder = bart_model.encoder
        self.decoder = bart_model.decoder

        self.copy_generator = CopyGenerator(input_dim=bart_model.config.d_model,
            hidden_dim_list=[512, 512, ontology_vocab_size])

    def forward(self, batch: TensorInputs):
        """Supports token_ids only."""
        # Encode inputs
        encoder_hidden_states = self.encoder(input_ids=batch.input_ids,
                                             attention_mask=batch.attn_mask).last_hidden_state

        outputs = []
        # Decode
        for step in range(1, self.max_seq_len - 1):
            decoder_hidden_states = self.decoder(input_ids=batch.semantic_parse_ids[:, :step],
                                                 atteention_mask=batch.semantic_parse_attn_mask[:, :step],
                                                 encoder_hidden_states=encoder_hidden_states,
                                                 encoder_attention_mask=batch.attn_mask).last_hidden_state
            decoder_hidden_states = decoder_hidden_states[:, -1:] # Get hidden_states of the last token

            # Copy or generate
            copy_probs = self.copy_generator(encoder_hidden_states, decoder_hidden_states, mode='train')

            outputs.append(copy_probs)
        return torch.stach(outputs, dim=1)

    def predict(self, batch: TensorInputs) -> Tensor:
        # Encode
        encoder_hidden_states = self.encoder(input_ids=batch.input_ids,
                                             attention_mask=batch.attn_mask).last_hidden_state

        # Init outputs with <BOS>
        outputs: Tensor = torch.tile(self.pad_token_id, (len(batch.input_ids), self.max_seq_len))
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
                encoder_attention_mask=batch.attn_mask.index_select(0, indices)).last_hidden_state
            decoder_hidden_states = decoder_hidden_states[:, -1:]   # Get hidden_states of the last token

            # Copy or generate
            # [batch_size, ontology_vocab_size + max_seq_len]
            outputs: Tensor = self.copy_generator(encoder_hidden_states.index_select(0, indices), decoder_hidden_states)

            # Update outputs
            outputs[indices] = torch.cat([past_outputs, outputs])

        return outputs
