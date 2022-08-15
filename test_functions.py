import unittest
from models.copy_generator import CopyGenerator

# Test models
class TestCopyGenerator(unittest.TestCase):
    def test_set_up (self) -> None:
        """Test setup CopyGenerator"""
        CopyGenerator(input_dim=768, hidden_dim_list=[768, 768, 24], num_heads=12, dropout=0.3)


if __name__ == '__main__':
    unittest.main()
