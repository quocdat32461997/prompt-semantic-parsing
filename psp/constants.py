from typing import NamedTuple, Optional, Dict, List, Union
from torch import Tensor


class Datasets:
    TOPv2: str = "/Users/datqngo/Desktop/projects/prompt_top/psp/dataset/TOPv2_Dataset"


class OntologyVocabs:
    TOPv2: str = "/Users/datqngo/Desktop/projects/prompt_top/psp/dataset/topv2_ontology_vocabs.pkl"


# TOPv2 dataset
TOPv2_DOMAIN_MAP: Dict[str, int] = {
    'alarm': 0,
    'event': 1,
    'messaging': 2,
    'music': 3,
    'navigation': 4,
    'reminder': 5,
    'weather': 6,
    'timer': 7,
}

class ListInputs(NamedTuple):
    domain: int
    utterance: str
    semantic_parse: str

class ParseInputs(NamedTuple):
    domain: Union[int, Tensor]
    input_ids: Tensor
    attn_mask: Tensor
    semantic_parse: Tensor
    semantic_parse_attn_mask: Tensor


class TensorInputs(NamedTuple):
    input_ids: Tensor
    attn_mask: Tensor
    inputs_embeds: Tensor
    semantic_parse_ids: Optional[Tensor]
    semantic_parse_attn_mask: Optional[Tensor]
    semantic_parse_embeds: Optional[Tensor]


PRETRAINED_BART_MODEL: str = 'facebook/bart-base'

ONTOLOGY_SCOPE_PATTERN: str = "\[|IN:\w+|SL:\w+|\]"
ONTOLOGY_PATTERN: str = "IN:\w+|SL:\w+"
