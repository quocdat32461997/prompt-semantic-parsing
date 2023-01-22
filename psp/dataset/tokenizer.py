import pickle
import torch
from typing import List, Dict, Union
from torch import Tensor
from transformers import BartTokenizer
from psp.constants import (
    OntologyVocabs,
    DatasetPaths,
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

    @property
    def intent_id_list(self) -> List[int]:
        return list(self.intent_to_id_map.values())

    @property
    def slot_id_list(self) -> List[int]:
        return list(self.slot_to_id_map.values())
