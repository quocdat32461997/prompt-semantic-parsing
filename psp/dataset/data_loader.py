import os
import pickle
from typing import List, Dict, Union
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import BartTokenizer
from psp.constants import OntologyVocabs, TOPv2_DOMAIN_MAP, ParseInputs, Datasets
from psp.dataset.data_utils import read_and_merge


class Tokenizer:
    def __init__(self, pretrained: str, dataset: str):
        # Init tokenizer and add ontology vocabs
        self.tokenizer: BartTokenizer = BartTokenizer.from_pretrained(pretrained)

        # Read onotlogy vocabs
        if dataset == Datasets.TOPv2:
            self._read_topv2_ontology_vocabs()
        else:
            raise ValueError("{} is an unsupported dataset.".format(dataset))

    def _read_topv2_ontology_vocabs(self):
        """Read TOPv2 ontology vocabs and add to tokenizer."""

        # Read ontology vocab
        with open(OntologyVocabs.TOPv2, 'rb') as file:
            self.ontology_per_domain_map: Dict[str, Dict[str, List[str]]] = pickle.load(file)

        # Get lists of intents and slots
        self.intent_list: List[str] = []
        self.slot_list: List[str] = []
        for ontology_per_domain in self.ontology_per_domain_map.values():
            self.intent_list.extend(ontology_per_domain['intents'])
            self.slot_list.extend(ontology_per_domain['slots'])

        # Remove duplicates
        self.intent_list = list(set(self.intent_list))
        self.slot_list = list(set(self.slot_list))

        # Add ontology vocabs to tokenizer
        self.ontology_list: List[str] = self.intent_list + self.slot_list
        self.tokenizer.add_tokens(self.ontology_list, special_tokens=True)

    def __call__(self, inputs: Union[str, List[str]], **kwargs) -> Union[List[int], List[List[int]]]:
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
    def ontology_vocab_size(self) -> int:
        return len(self.ontology_list)

    @property
    def num_intent(self) -> int:
        return len(self.intent_list)

    @property
    def num_slot(self) -> int:
        return len(self.slot_list)


class TOPv2Dataset(Dataset):
    BUCKET_DICT: Dict[str, str] = {
        'train': '_train.tsv',
        'eval': '_eval.tsv',
        'test': '_test.tsv',
    }

    def __init__(self, tokenizer: Tokenizer, bucket: str) -> None:
        super().__init__()

        # Read data
        self.data: pd.DataFrame = read_and_merge(
            [os.path.join(Datasets.TOPv2, domain + TOPv2Dataset.BUCKET_DICT[bucket]) for domain in TOPv2_DOMAIN_MAP.keys()])

        # Init tokenizer
        self.tokenizer: Tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.data)


class LowResourceTOpv2Dataset(TOPv2Dataset):
    def __getitem__(self, idx) -> ParseInputs:
        sample = self.data.iloc[idx]

        # Encode domain
        domain = TOPv2_DOMAIN_MAP[sample['domain']]

        # Tokenize utterance
        tokenized_utterance = self.tokenizer(
            sample['utterance'], padding='max_length', truncation=True, return_tensors='pt')

        # Tokenize semantic_parse
        tokenized_semantic_parse: List[int] = self.tokenizer(
            sample['semantic_parse'], padding='max_length', truncation=True, return_tensors='pt')

        return ParseInputs(domain=domain,
                           input_ids=tokenized_utterance['input_ids'],
                           attn_mask=tokenized_utterance['attention_mask'],
                           semantic_parse=tokenized_semantic_parse['input_ids'],
                           semantic_parse_attn_mask=tokenized_semantic_parse['attention_mask'])


class PromptTOPv2Dataset(TOPv2Dataset):
    def __getitem__(self, idx) -> ParseInputs:
        return None
