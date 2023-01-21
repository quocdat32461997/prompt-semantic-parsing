from typing import NamedTuple, Optional, Union
from enum import Enum
from torch import Tensor

# -100, the default ignorance token
IGNORED_INDEX: int = -100

# end of span token
EOSPAN_TOKEN: str = "]"


class RunMode(Enum):
    TRAIN: str = "train"
    EVAL: str = "eval"
    TEST: str = "test"


class DatasetPaths(Enum):
    TOPv2: str = "datasets/TOPv2_Dataset"
    TOP: str = "datasets/top-dataset-semantic-parsing"


class OntologyVocabs(Enum):
    TOPv2: str = "datasets/topv2_ontology_vocabs.pkl"
    TOP: str = "datasets/top_ontology_vocabs.pkl"


class TOPDomain(Enum):
    none: int = 0


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
    semantic_parse_ids: Tensor
    semantic_parse_attn_mask: Tensor
    intent_mask: Optional[Tensor] = None
    slot_mask: Optional[Tensor] = None
    ontology_token_mask: Optional[Tensor] = None


PRETRAINED_BART_MODEL: str = "facebook/bart-base"

ONTOLOGY_SCOPE_PATTERN: str = "\[IN:\w+|\[SL:\w+|\]"
ONTOLOGY_PATTERN: str = "\[IN:\w+|\[SL:\w+"
