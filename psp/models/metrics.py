import torch
from torchmetrics import Metric


class ExactMatch(Metric):
    """Exact Match Accuracy for the sentence-level. 
    By default, ignores tailing padding tokens for stricter accuracy.
    """

    def __init__(self, pad_token_id: int):
        super().__init__()
        self.pad_token_id: int = pad_token_id
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        preds, targets = self._input_format(preds, targets)
        assert preds.shape == targets.shape

        self.total += len(targets)

        for pred, target in zip(preds, targets):
            # Get index of the first PAD token in target
            first_pad_idx = torch.nonzero(target != self.pad_token_id, as_tuple=False).reshape([-1])
            first_pad_idx = -1 if len(first_pad_idx) == 0 else first_pad_idx[0]

            # Get target semantic parse
            target = target[:first_pad_idx]
            pred = pred[:first_pad_idx]

            self.correct += (pred == target).all()

    def compute(self):
        return self.correct.float() / self.total

# TO-DO: F1 and Acc. for all intents and slots using frequency

# TO-DO: labeled bracketing for intents, and intent-slots

# Tree validity - via bracket matching