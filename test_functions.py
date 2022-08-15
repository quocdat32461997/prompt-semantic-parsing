import unittest
from models import CopyGenerator, Seq2SeqCopyPointer

# Test models


class TestCopyGenerator(unittest.TestCase):
    def test_setup(self) -> None:
        """Test setup CopyGenerator"""
        CopyGenerator(input_dim=768, hidden_dim_list=[768, 768, 24], num_heads=12, dropout=0.3)

    def test_forward(self) -> None:
        pass


class TestSeq2SeqCopyPointer(unittest.TestCase):
    def test_setup(self) -> None:
        Seq2SeqCopyPointer(pretrained='facebook/bart-base', max_seq_len=128,
                           bos_token_id=1, eos_token_id=2, pad_token_id=0)

    def test_forward(self) -> None:
        pass

    def test_predict(self) -> None:
        pass


if __name__ == '__main__':
    unittest.main()
