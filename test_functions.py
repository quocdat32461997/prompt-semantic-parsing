from lib2to3.pgen2 import token
import unittest
from typing import List
from transformers import BartTokenizer

from psp.constants import PRETRAINED_BART_MODEL, Datasets, RunMode
from psp.models import CopyGenerator, Seq2SeqCopyPointer
from psp.dataset import (
    LowResourceTOpv2Dataset,
    Tokenizer,
    PromptTOPv2Dataset,
    DataLoader,
)

UTTERANCE: str = "Set alarm every minute for next hour"
TOKEN_IDS: List[int] = [0, 28512, 8054, 358, 2289, 13, 220, 1946, 2]
ATTN_MASK: List[int] = [1, 1, 1, 1, 1, 1, 1, 1, 1]
SEMANTIC_PARSE: str = "[IN:CREATE_ALARM Set alarm [SL:DATE_TIME_RECURRING every minute ] [SL:DURATION for next hour ] ]"

# Test models


class TestCopyGenerator(unittest.TestCase):
    def test_setup(self) -> None:
        """Test setup CopyGenerator"""
        CopyGenerator(
            input_dim=768, hidden_dim_list=[768, 768, 24], num_heads=12, dropout=0.3
        )

    def test_forward(self) -> None:
        pass


class TestSeq2SeqCopyPointer(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = Tokenizer(
            pretrained=PRETRAINED_BART_MODEL, dataset=Datasets.TOPv2
        )
        self.model = Seq2SeqCopyPointer(
            pretrained=PRETRAINED_BART_MODEL,
            ontology_vocab_size=self.tokenizer.ontology_vocab_size,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
        )

    def test_forward(self) -> None:
        pass

    def test_predict(self) -> None:
        pass


# Test tokenizer


class TestTokenizer(unittest.TestCase):
    def test_topv2_tokenization(self) -> None:
        tokenizer = Tokenizer(pretrained=PRETRAINED_BART_MODEL, dataset=Datasets.TOPv2)
        tokenized_outputs = tokenizer(UTTERANCE)

        # Test token-ids and attention_mask
        self.assertEqual(tokenized_outputs["input_ids"], TOKEN_IDS)
        self.assertEqual(tokenized_outputs["attention_mask"], ATTN_MASK)

    def test_adding_ontology_vocabs(self) -> None:
        topv2_tokenizer = Tokenizer(
            pretrained=PRETRAINED_BART_MODEL, dataset=Datasets.TOPv2
        )
        tokenizer = BartTokenizer.from_pretrained(PRETRAINED_BART_MODEL)

        # Original vocab size
        initial_vocab_size = len(tokenizer)
        new_vocab_size = topv2_tokenizer.vocab_size

        self.assertEqual(
            new_vocab_size - initial_vocab_size, topv2_tokenizer.ontology_vocab_size
        )


# Test datasets and dataloaders


class TestDataLoader(unittest.TestCase):
    def setUp(self) -> None:
        self.batch_size = 3

    def test_topv2_dataloader(self) -> None:
        """Test TOPv2-related datasets and dataloaders"""

        # Init tokenizer
        tokenizer = Tokenizer(pretrained=PRETRAINED_BART_MODEL, dataset=Datasets.TOPv2)

        # Init TOPv2-oriented dataloader
        data_loader = DataLoader(
            tokenizer=tokenizer,
            dataset_name=Datasets.TOPv2,
            dataset=LowResourceTOpv2Dataset(bucket=RunMode.TRAIN),
            batch_size=self.batch_size,
        )

        for batch in data_loader:
            self.assertEqual(len(batch.domain), self.batch_size)
            self.assertEqual(
                list(batch.input_ids.shape), [self.batch_size, tokenizer.max_seq_len]
            )
            break


if __name__ == "__main__":
    unittest.main()
