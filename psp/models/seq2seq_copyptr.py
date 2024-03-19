import torch
from torch import Tensor
from transformers import BartModel
from typing import List

from psp.models.pointer_generator import PointerGenerator
from psp.constants import ParseInputs, RunMode
from psp.models.decoding_utils import BeamSearch
from psp.models.model_utils import get_arange_matrix

class Seq2SeqCopyPointer(torch.nn.Module):
    def __init__(
        self,
        pretrained: str,
        vocab_size: int,
        bos_token_id: int,
        eos_token_id: int,
        pad_token_id: int,
        beam_size: int,
        alpha: float,
        reward: float,
        max_dec_steps: int = None,
        dropout: float = 0.1,
        **kwargs,
    ) -> None:
        super(Seq2SeqCopyPointer, self).__init__()

        # Load pretrained model
        bart_model = BartModel.from_pretrained(pretrained)

        # resize size of token-embedding
        bart_model.resize_token_embeddings(vocab_size)

        assert bos_token_id == bart_model.config.bos_token_id
        assert eos_token_id == bart_model.config.eos_token_id
        assert pad_token_id == bart_model.config.pad_token_id

        self.model_config = bart_model.config
        self.eos_token_id: int = eos_token_id
        self.pad_token_id: int = pad_token_id
        self.bos_token_id: int = bos_token_id

        self.max_dec_steps: int = (
            max_dec_steps if max_dec_steps else self.model_config.max_position_embeddings
        )
        assert self.max_dec_steps <= self.model_config.max_position_embeddings

        self.encoder = bart_model.encoder
        self.decoder = bart_model.decoder

        self.searcher: BeamSearch = BeamSearch(
            beam_size=beam_size,
            alpha=alpha,
            reward=reward,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
            max_queue_size=kwargs["max_queue_size"],
            min_dec_steps=kwargs["min_dec_steps"],
            n_best=kwargs["n_best"],
            max_seq_len=self.max_dec_steps,
        )
    
    def _encode(self, input_ids: Tensor, attn_mask: Tensor):
        """
        Args:
            - input_ids: Tensor of word-ids, [batch_size, max_seq_len]
            - attn_mask: Tensor of attention masks, [batch_size, max_seq_len]
        Returns:
            Last hidden-state of the encoder in shape [batch_size, max_seq_len, embed_dim]
        """
        return self.encoder(
            input_ids=input_ids, attention_mask=attn_mask
        ).last_hidden_state

    def forward(self, batch: ParseInputs, run_mode: RunMode) -> Tensor:
        return (
            self._forward(batch) if run_mode == RunMode.TRAIN else self._generate(batch)
        )

class Seq2SeqVocabCopyPointer(Seq2SeqCopyPointer):
    def __init__(
        self,
        pretrained: str,
        vocab_size: int,
        ontology_vocab_ids: List[int],
        bos_token_id: int,
        eos_token_id: int,
        pad_token_id: int,
        beam_size: int,
        alpha: float,
        reward: float,
        max_dec_steps: int = None,
        dropout: float = 0.1,
        **kwargs,
    ):
        super(Seq2SeqVocabCopyPointer, self).__init__(pretrained, vocab_size, bos_token_id, eos_token_id, pad_token_id, beam_size, alpha, reward, max_dec_steps, dropout, **kwargs)
        
        self.pointer_generator: PointerGenerator = PointerGenerator(
            vocab_size=vocab_size,
            ontology_vocab_ids=ontology_vocab_ids,
            input_dim=self.model_config.d_model,
            gen_hidden_dim_list=[512, 512, len(ontology_vocab_ids)],
            dropout=dropout,
        )
    
    def _forward(self, batch: ParseInputs) -> Tensor:
        """Supports token_ids only."""
        # Encode inputs
        encoder_hidden_states = self._encode(
            input_ids=batch.input_ids, attn_mask=batch.attn_mask
        )  # [batch_size, max_seq_len, embed_dim]

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
            source_input_ids=batch.input_ids,
            encoder_attn_mask=batch.attn_mask,
            encoder_hidden_states=encoder_hidden_states,
            decoder_hidden_states=decoder_hidden_states,
        )
        return vocab_probs

    def _generate(self, batch: ParseInputs) -> Tensor:
        batch_size: int = len(batch.input_ids)

        # Encode
        encoder_hidden_states = self._encode(
            input_ids=batch.input_ids, attn_mask=batch.attn_mask
        )  # [batch_size, encoder_input_len, embed_dim]

        # Initialize decoder inputs
        self.searcher.init_decoder_inputs(
            batch_size=batch_size,
            device=encoder_hidden_states.device,
        )

        # Decode
        # First index is reserved for <BOS>
        for step in range(1, self.max_dec_steps):

            # Stop generation
            if self.searcher.is_search_done():
                break

            decoder_inputs, decoder_attn_mask = self.searcher.get_decoder_inputs(
                step=step, device=encoder_hidden_states.device,
            )
            decoder_hidden_states = self.decoder(
                input_ids=decoder_inputs,
                attention_mask=decoder_attn_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=batch.attn_mask,
            ).last_hidden_state

            # Get hidden_states of the last token
            decoder_hidden_states = decoder_hidden_states[:, -1:]

            # Generate probs to copy from source tokens or generate ontology tokens
            # [batch_size or batch_size * beam_size, 1, vocab_size]
            vocab_probs: Tensor = self.pointer_generator(
                source_input_ids=batch.input_ids,
                encoder_attn_mask=batch.attn_mask,
                encoder_hidden_states=encoder_hidden_states,
                decoder_hidden_states=decoder_hidden_states,
                run_mode=RunMode.EVAL,
            )

            # beam-search
            vocab_probs = vocab_probs.view((batch_size, -1))
            self.searcher(probs=vocab_probs)

        # squeeze and return final outputs
        return self.searcher.get_final_outputs().to(encoder_hidden_states.device)


class Seq2SeqIndexCopyPointer(Seq2SeqCopyPointer):
    def __init__(
        self,
        pretrained: str,
        vocab_size: int,
        output_vocab_size: int,
        ontology_vocab_ids: List[int],
        bos_token_id: int,
        eos_token_id: int,
        pad_token_id: int,
        beam_size: int,
        alpha: float,
        reward: float,
        max_dec_steps: int = None,
        dropout: float = 0.1,
        **kwargs,
    ):
        super(Seq2SeqIndexCopyPointer, self).__init__(pretrained, vocab_size, bos_token_id, eos_token_id, pad_token_id, beam_size, alpha, reward, max_dec_steps, dropout, **kwargs)

        self.pointer_generator: PointerGenerator = PointerGenerator(
            vocab_size=output_vocab_size,
            ontology_vocab_ids=ontology_vocab_ids,
            input_dim=self.config.d_model,
            gen_hidden_dim_list=[512, 512, len(ontology_vocab_ids)],
            dropout=dropout,
        )

    def _forward(self, batch: ParseInputs) -> Tensor:
        """Supports token_ids only."""
        # Encode inputs
        encoder_hidden_states = self._encode(
            input_ids=batch.input_ids, attn_mask=batch.attn_mask
        )  # [batch_size, max_seq_len, embed_dim]

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
            source_input_ids=get_arange_matrix(batch.input_ids.shape).to(encoder_hidden_states.device),
            encoder_attn_mask=batch.attn_mask,
            encoder_hidden_states=encoder_hidden_states,
            decoder_hidden_states=decoder_hidden_states,
        )
        return vocab_probs

    def _generate(self, batch: ParseInputs) -> Tensor:
        batch_size: int = len(batch.input_ids)

        # Encode
        encoder_hidden_states = self._encode(
            input_ids=batch.input_ids, attn_mask=batch.attn_mask
        )  # [batch_size, encoder_input_len, embed_dim]

        # Initialize decoder inputs
        self.searcher.init_decoder_inputs(
            batch_size=batch_size,
            device=encoder_hidden_states.device,
        )

        # Decode
        # First index is reserved for <BOS>
        for step in range(1, self.max_dec_steps):

            # Stop generation
            if self.searcher.is_search_done():
                break

            decoder_inputs, decoder_attn_mask = self.searcher.get_decoder_inputs(
                step=step, device=encoder_hidden_states.device,
            )
            decoder_hidden_states = self.decoder(
                input_ids=decoder_inputs,
                attention_mask=decoder_attn_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=batch.attn_mask,
            ).last_hidden_state

            # Get hidden_states of the last token
            decoder_hidden_states = decoder_hidden_states[:, -1:]

            # Generate probs to copy from source tokens or generate ontology tokens
            # [batch_size or batch_size * beam_size, 1, vocab_size]
            vocab_probs: Tensor = self.pointer_generator(
                source_input_ids=get_arange_matrix(batch.input_ids.shape).to(encoder_hidden_states.device),
                encoder_attn_mask=batch.attn_mask,
                encoder_hidden_states=encoder_hidden_states,
                decoder_hidden_states=decoder_hidden_states,
                run_mode=RunMode.EVAL,
            )

            # beam-search
            vocab_probs = vocab_probs.view((batch_size, -1))
            self.searcher(probs=vocab_probs)

        # squeeze and return final outputs
        return self.searcher.get_final_outputs().to(encoder_hidden_states.device)