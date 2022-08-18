from typing import NamedTuple, Optional, Union
from enum import Enum
from torch import Tensor


class RunMode(Enum):
    TRAIN: str = 'train'
    EVAL: str = 'eval'
    TEST: str = 'test'


class Datasets(Enum):
    TOPv2: str = "/Users/datqngo/Desktop/projects/prompt-semantic-parsing/datasets/TOPv2_Dataset"


class OntologyVocabs(Enum):
    TOPv2: str = "/Users/datqngo/Desktop/projects/prompt-semantic-parsing/datasets/topv2_ontology_vocabs.pkl"


class TOPv2Domain(Enum):
    alarm: int = 0
    event: int = 1
    messaging: int = 2
    music: int = 3
    navigation: int = 4
    reminder: int = 5
    weather: int = 6
    timer: int = 7


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
