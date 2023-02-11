import torch
from torch import Tensor
from typing import Dict, List, Tuple

from psp.constants import (
    IGNORED_INDEX,
    ParseOutputs,
    ONTOLOGY_TYPE_LIST,
    SUB_ONTOLOGY_TYPE_LIST,
)

class ParseTransform(torch.nn.Module):
    """
    Map tokens to corresponding classes of intents, slots, and the closing ontology token ]
    """

    def __init__(
        self,
        intent_id_list: List[int],
        slot_id_list: List[int],
        ontology_id_list: List[int],
        vocab_size: int,
        device: str = 'cpu',
    ) -> None:
        super(ParseTransform, self).__init__()

        # Fill in classes for intents and slots
        self.vocab_tensor: Tensor = torch.full((vocab_size,), IGNORED_INDEX)
        self.vocab_tensor[intent_id_list] = torch.arange(len(intent_id_list))
        self.vocab_tensor[slot_id_list] = torch.arange(len(slot_id_list))
        # Move to same device
        self.vocab_tensor = self.vocab_tensor.to(device) 

        # Mappings of ontology tokens, intents, and slots
        self.ontology_maps: Dict[str, Tensor] = {
            "ontology": torch.tensor(ontology_id_list).to(device),
            "intents": torch.tensor(intent_id_list).to(device),
            "slots": torch.tensor(slot_id_list).to(device),
        }
        assert list(self.ontology_maps.keys()) == ONTOLOGY_TYPE_LIST

        # Mappings of oov tokens
        self.oov_token_maps: Dict[str, Tensor] = {
            "intents": torch.tensor(len(intent_id_list) + 1).to(device),
            "slots": torch.tensor(len(slot_id_list) + 1).to(device),
        }
        assert list(self.oov_token_maps.keys()) == SUB_ONTOLOGY_TYPE_LIST

    def retrieve(
        self, outputs: Tensor, targets: Tensor, token_type: str
    ) -> Tuple[Tensor, Tensor]:
        assert outputs.shape == targets.shape

        # Get valid tokens
        valid_tokens: Tensor = self.ontology_maps[token_type]#.to(targets.device)

        # Get mask of valid tokens from outputs
        mask = torch.isin(targets, valid_tokens)

        # Retrieve valid tokens in outputs and targets
        outputs = torch.masked_select(outputs, mask)
        targets = torch.masked_select(targets, mask)

        return outputs, targets

    def transform(self, outputs: Tensor, targets: Tensor, token_type: str) -> Tensor:
        """
        If token_type is intents or slots
            Map raw token-ids of intents and slots to corresponding classes
            For example:
                valid_tokens = [1,2,3]
                class_tensor = [0, 1, 2]
                oov_token = 4

                outputs = [1, 2, 0, 100, 5, 3]
                outputs = [0, 1, 4, 4, 4, 2] (final)
        Else:
            Perform exact-match comparison
            And convert to one-hot mappings
            For example:
                outputs = [1, 2, 0, 100, 5, 3]
                targets = [1, 3, 1, 0, 5, 3]

                outputs = [1, 0, 0, 0, 1, 1]
                targets = [1, 1, 1, 1, 1, 1]

        Args:
            - inputs: 1D Tensor of tokens
        Returns:
            - outputs: 1D Tensor of classes
        """
        if token_type in SUB_ONTOLOGY_TYPE_LIST:

            # Retrieve valid_tokens and the corresponding oov_token
            valid_tokens: Tensor = self.ontology_maps[token_type]
            oov_token: Tensor = self.oov_token_maps[token_type]

            # map tokens to classes
            outputs: Tensor = self.vocab_tensor[outputs]

            # Map OOV tokens to their unique id
            oov_mask: Tensor = torch.logical_not(torch.isin(outputs, valid_tokens))
            outputs = outputs.masked_fill(oov_mask, oov_token)
        else:
            # Convert exact-match comparison to one-hot values
            outputs = (outputs == targets).to(torch.int)
            targets = torch.ones_like(targets)

        return outputs, targets

    def forward(self, outputs: Tensor, targets: Tensor, token_type: str) -> Tensor:
        assert token_type in ONTOLOGY_TYPE_LIST

        # Retrive ontology tokens, intents, and slots only
        outputs, targets = self.retrieve(outputs, targets, token_type)

        # Perform transformation
        outputs, targets = self.transform(outputs, targets, token_type)

        return ParseOutputs(outputs=outputs, targets=targets)
