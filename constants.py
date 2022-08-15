from typing import NamedTuple, Optional
from torch import Tensor


class TensorInputs(NamedTuple):
    input_ids: Optional[Tensor]
    attn_mask: Optional[Tensor]
    inputs_embeds: Optional[Tensor]
