import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from psp.dataset.tokenizer import Tokenizer, PointerTokenizer
from psp.constants import DatasetPaths, ListInputs, ParseInputs
from typing import Dict, List

class InputTransform:
    def __init__(self, pretrained: str, dataset_path: DatasetPaths) -> None:
        self.tokenizer: Tokenizer = None
        pass

    def __call__(self, batch: List[ListInputs]) -> ParseInputs:
        pass

class TokenTransform(InputTransform):
    def __init__(self, pretrained: str, dataset_path: DatasetPaths, use_pointer_data: bool = False) -> None:
        super(TokenTransform, self).__init__(pretrained, dataset_path)

        self.use_pointer_data: bool = use_pointer_data
        self.tokenizer: Tokenizer = Tokenizer(pretrained=pretrained, dataset_path=dataset_path)
    
    def _pad_sequence(self, inputs: List[Tensor]) -> Tensor:
        outputs: Tensor = pad_sequence(inputs, batch_first=True, padding_value=self.tokenizer.pad_token_id)

        if outputs.shape[-1] > self.tokenizer.max_seq_len:
            # Trim to max_seq_len
            outputs = outputs[..., :self.tokenizer.max_seq_len]

            # Preserve <eos> token_id
            outputs[..., -1] = torch.where(outputs[..., -1] == self.tokenizer.eos_token_id, outputs, self.tokenizer.pad_token_id)

        return outputs

    def _get_attention_mask(self, inputs: Tensor) -> Tensor:
        return torch.where(inputs != self.tokenizer.pad_token_id, 1, 0)

    def __call__(self, batch: List[ListInputs]) -> ParseInputs:
        """Custom collate function to batch ParseInputs"""
        domain_list: List[int] = []
        utterance_list: List[str] = []
        semantic_parse_list: List[str] = []
        pointer_parse_list: List[str] = []
        
        # Get inptus
        for inputs in batch:
            domain_list.append(inputs.domain)
            utterance_list.append(inputs.utterance)
            semantic_parse_list.append(inputs.semantic_parse)

            if self.use_pointer_data and inputs.pointer_parse:
                pointer_parse_list.append(inputs.pointer_parse)

        assert self.use_pointer_data and pointer_parse_list

        domain_tensor: Tensor = torch.tensor(domain_list)
        
        # Pad utterance-, semantic- or pointer-parse
        token_tensor: Tensor = self._pad_sequence(utterance_list)
        semantic_parse_tensor: Tensor = self._pad_sequence(pointer_parse_list) if self.use_pointer_data and pointer_parse_list else self._pad_sequence(semantic_parse_list)

        # Get attention mask
        attn_mask = self._get_attention_mask(token_tensor)
        semantic_parse_attn_mask = self._get_attention_mask(semantic_parse_tensor)

        return ParseInputs(
            domain=domain_tensor,
            input_ids=token_tensor,
            attn_mask=attn_mask.to(torch.float),
            semantic_parse_ids=semantic_parse_tensor,
            semantic_parse_attn_mask=semantic_parse_attn_mask,
        )

class TextTransform(InputTransform):
    
    def __init__(self, pretrained: str, dataset_path: DatasetPaths) -> None:
        super(TextTransform, self).__init__(pretrained, dataset_path)

        self.tokenizer: Tokenizer = PointerTokenizer(pretrained=pretrained, dataset_path=dataset_path)

    def __call__(self, batch: List[ListInputs]) -> ParseInputs:
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

        return ParseInputs(
            domain=domain_tensor,
            input_ids=tokenized_utterance["input_ids"],
            attn_mask=tokenized_utterance["attention_mask"].to(torch.float),
            semantic_parse_ids=tokenized_semantic_parse["input_ids"],
            semantic_parse_attn_mask=tokenized_semantic_parse["attention_mask"],
            pointer_parse_ids=None
        )