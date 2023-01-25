import torch
from typing import List, Optional
from torch import Tensor
from torch.utils.data import DataLoader
from psp.constants import (
    ListInputs,
    ParseInputs,
    DatasetPaths,
    RunMode,
)
from .tokenizer import Tokenizer


class SMPDataLoader(DataLoader):
    def __init__(
        self,
        tokenizer: Tokenizer,
        dataset_path: DatasetPaths,
        run_mode: RunMode = RunMode.TRAIN,
        **kwargs
    ):
        # Get collate_fn
        collate_fn = None
        if dataset_path == DatasetPaths.TOPv2:
            collate_fn = self.collate_topv2_parse_inputs

        super().__init__(collate_fn=collate_fn, **kwargs)

        self.tokenizer: Tokenizer = tokenizer
        self.run_mode: RunMode = run_mode

    def collate_topv2_parse_inputs(self, batch: List[ListInputs]) -> ParseInputs:
        """Custom collate function to batch ParseInputs"""
        domain_list: List[int] = []
        utterance_list: List[str] = []
        semantic_parse_list: List[str] = []

        # Get inptus
        for inputs in batch:
            domain_list.append(inputs.domain)
            utterance_list.append(inputs.utterance)
            semantic_parse_list.append(inputs.semantic_parse)

        domain_tensor: Tensor = torch.tensor(domain_list)

        # Tokenize utterance and semantic_parse
        tokenized_utterance = self.tokenizer.batch_encode_plus(
            utterance_list,
            truncation=True,
            add_special_tokens=True,
            max_length=self.tokenizer.max_seq_len,
            padding="longest",
            return_tensors="pt",
        )
        tokenized_semantic_parse = self.tokenizer.batch_encode_plus(
            semantic_parse_list,
            truncation=True,
            add_special_tokens=True,
            max_length=self.tokenizer.max_seq_len,
            padding="longest",
            return_tensors="pt",
        )

        # intent_mask: Optional[Tensor] = None
        # slot_mask: Optional[Tensor] = None
        # ontology_token_mask: Optional[Tensor] = None
        if self.run_mode == RunMode.EVAL:
            """
            # Initialize dummpy tensors
            one_tensor = torch.ones_like(tokenized_semantic_parse["input_ids"])
            zero_tensor = torch.zeros_like(tokenized_semantic_parse["input_ids"])

            intent_mask = torch.where(
                torch.isin(
                    tokenized_semantic_parse["input_ids"], self.tokenizer.intent_tensors
                ),
                one_tensor,
                zero_tensor,
            )
            slot_mask = torch.where(
                torch.isin(
                    tokenized_semantic_parse["input_ids"], self.tokenizer.slot_tensors
                ),
                one_tensor,
                zero_tensor,
            )
            eospan_token_mask = torch.where(
                torch.isin(
                    tokenized_semantic_parse["input_ids"],
                    self.tokenizer.eospan_token_id,
                ),
                one_tensor,
                zero_tensor,
            )
            ontology_token_mask = intent_mask + slot_mask + eospan_token_mask
            """
            pass

        # Convert to Tensor and parse back into ParseInputs
        return ParseInputs(
            domain=domain_tensor,
            input_ids=tokenized_utterance["input_ids"],
            attn_mask=tokenized_utterance["attention_mask"],
            semantic_parse_ids=tokenized_semantic_parse["input_ids"],
            semantic_parse_attn_mask=tokenized_semantic_parse["attention_mask"],
            # intent_mask=intent_mask,
            # slot_mask=slot_mask,
            # ontology_token_mask=ontology_token_mask,
        )
