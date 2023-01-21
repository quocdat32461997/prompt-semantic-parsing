import pickle
import torch
from typing import List, Dict, Union, Optional
from torch import Tensor
from torch.utils.data import DataLoader
from transformers import BartTokenizer
from psp.constants import (
    ListInputs,
    OntologyVocabs,
    ParseInputs,
    DatasetPaths,
    RunMode,
    EOSPAN_TOKEN,
)


class Tokenizer:
    """] is the special token to indicate the enclosure of a span (either by intent or slot)"""

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
        self.slot_list = list(set(self.slot_list))

        # Add ontology vocabs to tokenizer
        # ] is the special token indicating the enclousre of a span
        ontology_list: List[str] = list(
            set(self.intent_list + self.slot_list + [EOSPAN_TOKEN])
        )

        new_added_ontology_token_num: int = self.tokenizer.add_tokens(
            ontology_list, special_tokens=True
        )
        print("Added {} ontology tokens.".format(new_added_ontology_token_num))

        # get ids of ontology vocab
        self.ontology_id_list: List[int] = self.tokenizer.encode(ontology_list)[1:-1]
        ontology_to_id_map: Dict[str, int] = {
            key: value for key, value in zip(ontology_list, self.ontology_id_list)
        }

        # token_id of EOSPAN_TOKEN
        self.eospan_token_id: int = ontology_to_id_map[EOSPAN_TOKEN]

        # create mappings: ontology -> ids and ids -> ontology
        self.intent_to_id_map: Dict[str, int] = {}
        self.id_to_intent_map: Dict[int, str] = {}

        for key in self.intent_list:
            value = ontology_to_id_map[key]
            self.intent_to_id_map[key] = value
            self.id_to_intent_map[value] = key

        self.slot_to_id_map: Dict[str, int] = {}
        self.id_to_slot_map: Dict[int, str] = {}
        for key in self.slot_list:
            value = ontology_to_id_map[key]
            self.slot_to_id_map[key] = value
            self.id_to_slot_map[value] = key

        # Save tensors of intents and slots
        self.intent_tensors: Tensor = torch.tensor(list(self.intent_to_id_map.values()))
        self.slot_tensors: Tensor = torch.tensor(list(self.slot_to_id_map.values()))

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

    @property
    def map_id_to_intent(self, id: int) -> str:
        return self.id_to_intent_map[id]

    @property
    def map_id_to_slot(self, id: int) -> str:
        return self.id_to_slot_map[id]

    @property
    def map_intent_to_id(self, key: str) -> int:
        return self.map_intent_to_id[key]

    @property
    def map_slot_to_id(self, key: str) -> int:
        return self.map_slot_to_id[key]

    @property
    def end_of_span_token(self):
        return EOSPAN_TOKEN

    @property
    def end_of_span_token_id(self):
        return self.eospan_token_id


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

        intent_mask: Optional[Tensor] = None
        slot_mask: Optional[Tensor] = None
        ontology_token_mask: Optional[Tensor] = None
        if self.run_mode == RunMode.EVAL:
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

        # Convert to Tensor and parse back into ParseInputs
        return ParseInputs(
            domain=domain_tensor,
            input_ids=tokenized_utterance["input_ids"],
            attn_mask=tokenized_utterance["attention_mask"],
            semantic_parse_ids=tokenized_semantic_parse["input_ids"],
            semantic_parse_attn_mask=tokenized_semantic_parse["attention_mask"],
            intent_mask=intent_mask,
            slot_mask=slot_mask,
            ontology_token_mask=ontology_token_mask,
        )
