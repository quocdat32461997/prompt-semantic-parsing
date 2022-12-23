import pickle
import torch
from typing import List, Dict, Union
from torch import Tensor
from torch.utils.data import DataLoader
from transformers import BartTokenizer
from psp.constants import (
    ListInputs,
    OntologyVocabs,
    ParseInputs,
    DatasetPaths,
)


class Tokenizer:
    def __init__(self, pretrained: str, dataset_path: str):
        # Init tokenizer and add ontology vocabs
        self.tokenizer: BartTokenizer = BartTokenizer.from_pretrained(pretrained)
        # Read onotlogy vocabs
        if dataset_path == DatasetPaths.TOPv2:
            self._read_topv2_ontology_vocabs()
        else:
            raise ValueError("{} is an unsupported dataset.".format(dataset_path))

    def batch_encode_plus(self, batch_text: List[str], **kwargs):
        return self.tokenizer.batch_encode_plus(batch_text, **kwargs)

    def _read_topv2_ontology_vocabs(self):
        """Read TOPv2 ontology vocabs and add to tokenizer."""
        # Read ontology vocab
        with open(OntologyVocabs.TOPv2.value, "rb") as file:
            self.ontology_per_domain_map: Dict[str, Dict[str, List[str]]] = pickle.load(
                file
            )
        # Get lists of intents and slots
        self.intent_list: List[str] = []
        self.slot_list: List[str] = []
        for ontology_per_domain in self.ontology_per_domain_map.values():
            self.intent_list.extend(ontology_per_domain["intents"])
            self.slot_list.extend(ontology_per_domain["slots"])
        # Remove duplicates (if existed)
        self.intent_list = list(set(self.intent_list))

        # Add ontology vocabs to tokenizer
        self.ontology_list: List[str] = list(set(self.intent_list + self.slot_list))

        new_added_ontology_tokens = self.tokenizer.add_tokens(
            self.ontology_list, special_tokens=True
        )
        print("Added {} ontology tokens.".format(new_added_ontology_tokens))

        # get ids of ontology vocab
        self.ontology_id_list = self.tokenizer.encode(self.ontology_list)[1:-1]

    def __call__(
        self, inputs: Union[str, List[str]], **kwargs
    ) -> Union[List[int], List[List[int]]]:
        return self.tokenizer(inputs, **kwargs)

    @property
    def max_seq_len(self) -> int:
        return self.tokenizer.model_max_length

    @property
    def bos_token_id(self) -> int:
        return self.tokenizer.bos_token_id

    @property
    def eos_token_id(self) -> int:
        return self.tokenizer.eos_token_id

    @property
    def pad_token_id(self) -> int:
        return self.tokenizer.pad_token_id

    @property
    def vocab(self) -> Dict[str, int]:
        return self.tokenizer.get_vocab()

    @property
    def vocab_size(self) -> int:
        return len(self.tokenizer)

    @property
    def ontology_vocab_ids(self) -> int:
        return self.ontology_id_list

    @property
    def ontology_vocab_size(self) -> int:
        return len(self.ontology_list)

    @property
    def num_intent(self) -> int:
        return len(self.intent_list)

    @property
    def num_slot(self) -> int:
        return len(self.slot_list)


class SMPDataLoader(DataLoader):
    def __init__(self, tokenizer: Tokenizer, dataset_path: DatasetPaths, **kwargs):
        # Get collate_fn
        collate_fn = None
        if dataset_path == DatasetPaths.TOPv2:
            collate_fn = self.collate_topv2_parse_inputs

        super().__init__(collate_fn=collate_fn, **kwargs)

        self.tokenizer: Tokenizer = tokenizer

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

        # Convert to Tensor and parse back into ParseInputs
        return ParseInputs(
            domain=domain_tensor,
            input_ids=tokenized_utterance["input_ids"],
            attn_mask=tokenized_utterance["attention_mask"],
            semantic_parse_ids=tokenized_semantic_parse["input_ids"],
            semantic_parse_attn_mask=tokenized_semantic_parse["attention_mask"],
        )
