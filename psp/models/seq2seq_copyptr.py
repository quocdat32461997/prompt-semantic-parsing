import torch
from torch import Tensor
from transformers import BartModel
from typing import List

from psp.models.pointer_generator import PointerGenerator
from psp.constants import ParseInputs, RunMode
from psp.models.decoding_utils import BeamSearch


class Seq2SeqCopyPointer(torch.nn.Module):
    def __init__(
        self,
        pretrained: str,
        vocab_size: int,
        ontology_vocab_ids: List[int],
        bos_token_id: int,
        eos_token_id: int,
        pad_token_id: int,
    ):
        super().__init__()
        bart_model = BartModel.from_pretrained(pretrained)

        # resize size of token-embedding
        bart_model.resize_token_embeddings(vocab_size)

        assert bos_token_id == bart_model.config.bos_token_id
        assert eos_token_id == bart_model.config.eos_token_id
        assert pad_token_id == bart_model.config.pad_token_id

        self.eos_token_id: int = eos_token_id
        self.pad_token_id: int = pad_token_id
        self.bos_token_id: int = bos_token_id

        self.max_seq_len: int = bart_model.config.max_position_embeddings

        self.encoder = bart_model.encoder
        self.decoder = bart_model.decoder

        self.pointer_generator = PointerGenerator(
            vocab_size=vocab_size,
            ontology_vocab_ids=ontology_vocab_ids,
            input_dim=bart_model.config.d_model,
            hidden_dim_list=[512, 512, len(ontology_vocab_ids)],
        )

        self.searcher: BeamSearch = BeamSearch()

    def forward(self, batch: ParseInputs) -> Tensor:
        """Supports token_ids only."""
        # Encode inputs
        encoder_hidden_states = self.encoder(
            input_ids=batch.input_ids, attention_mask=batch.attn_mask
        ).last_hidden_state  # [batch_size, max_seq_len, embed_dim]

        decoder_hidden_states_list: List[Tensor] = []

        # Decode
        decoder_seq_len = batch.semantic_parse_ids.shape[-1]
        for step in range(1, decoder_seq_len):
            decoder_hidden_states = self.decoder(
                input_ids=batch.semantic_parse_ids[:, :step],
                attention_mask=batch.semantic_parse_attn_mask[:, :step],
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=batch.attn_mask,
            ).last_hidden_state
            # Get hidden_states of the last token
            decoder_hidden_states_list.append(decoder_hidden_states[:, -1:])
        decoder_hidden_states: Tensor = torch.concat(decoder_hidden_states_list, dim=1)

        # Get probs to copy or generate
        vocab_probs: Tensor = self.pointer_generator(
            source_input_ids=batch.input_ids.clone(),
            encoder_hidden_states=encoder_hidden_states,
            decoder_hidden_states=decoder_hidden_states,
        )
        # Apply log over vocab_probs
        vocab_probs = torch.log(vocab_probs)

        return vocab_probs

    def predict(self, batch: ParseInputs) -> Tensor:
        # Encode
        encoder_hidden_states = self.encoder(
            input_ids=batch.input_ids, attention_mask=batch.attn_mask
        ).last_hidden_state

        # Init outputs with <BOS>
        outputs: Tensor = torch.tile(
            self.pad_token_id, (len(batch.input_ids), self.max_seq_len)
        )
        outputs[:, 0] = self.bos_token_id

        # Decode
        for _ in range(self.max_max_seq_len):
            # Get non-terminal outputs
            indices = (outputs[:, -1] != self.eos_token_id).int() * (
                outputs[:, -1] != self.pad_token_id
            ).int()
            indices = torch.nonzero(indices).reshape(-1)  # shape = [-1] or empty

            # Stop if meeting EOS or PAD
            if len(indices) == 0:
                break
            past_outputs = torch.index_select(outputs, 0, indices)

            decoder_hidden_states = self.decoder(
                input_ids=past_outputs,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=batch.attn_mask.index_select(0, indices),
            ).last_hidden_state
            # Get hidden_states of the last token
            decoder_hidden_states = decoder_hidden_states[:, -1:]

            # Generate probs to copy from source tokens or generate ontology tokens
            # [batch_size, 1, vocab_size]
            vocab_probs: Tensor = self.pointer_generator(
                source_input_ids=batch.input_ids.index_select(0, indices).clone(),
                encoder_hidden_states=encoder_hidden_states.index_select(0, indices),
                encoder_attn_mask=batch.attn_mask.index_select(0, indices),
                decoder_hidden_states=decoder_hidden_states,
                run_mode=RunMode.TEST,
            )

            # beam-search
            outputs = self.searcher(outputs, past_outputs)

            # Update outputs
            outputs[indices] = torch.cat([past_outputs, outputs])

        return outputs
