from typing import NamedTuple, Optional, Union, List, Dict
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
    utterance: Union[str, Tensor]
    semantic_parse: Union[str, Tensor]
    pointer_parse: Union[str, Tensor] = None


class ParseInputs(NamedTuple):
    domain: Union[int, Tensor]
    input_ids: Tensor
    attn_mask: Tensor
    semantic_parse_ids: Tensor
    semantic_parse_attn_mask: Tensor


class ParseOutputs(NamedTuple):
    outputs: Tensor
    targets: Tensor


PRETRAINED_BART_MODEL: str = "facebook/bart-base"

ONTOLOGY_SCOPE_PATTERN: str = "\[IN:\w+|\[SL:\w+|\]"
ONTOLOGY_PATTERN: str = "\[IN:\w+|\[SL:\w+"
TIME_SPACED_PATTERN: str = "\d{1,2}\s*(\:|\.)?\s*\d{0,2}\s*(([aA]|[pP])[mM]?)?(?!\w+)" # time format ending with space or non-alphanumeric
SHORT_FORM_SPACED_PATTERN : str = "\w+\s+'([rR][eE]|[sS]|[dD])?(?!\w+)"
MULTI_WHITESPACE_PATTERN: str = "\s+"
SINGLE_SPACE: str = " "

ONTOLOGY_TYPE_LIST: List[str] = ["ontology", "intents", "slots"]
SUB_ONTOLOGY_TYPE_LIST: List[str] = ["intents", "slots"]
