import torch
from torchmetrics import Metric, F1Score, Precision, Recall
from torch import Tensor
from psp.constants import IGNORED_INDEX, ONTOLOGY_TYPE_LIST, ParseOutputs


class ExactMatch(Metric):
    """Exact Match Accuracy for the sentence-level.
    By default, ignores tailing padding tokens for stricter accuracy.
    """

    def __init__(self) -> None:
        super(ExactMatch, self).__init__()
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, targets: Tensor) -> None:
        assert preds.shape == targets.shape

        seq_len = preds.shape[-1]
        self.total += len(targets)

        # Comparing
        outputs = (preds == targets).sum(dim=-1)
        self.correct += (outputs == seq_len).sum()

    def compute(self) -> Tensor:
        return {"em_acc": self.correct.float() / self.total}


class IntentSlotMatch(Metric):
    """
    Compute:
        - Acc. Detection of ontology
        - Acc. Detection of intents
        - Metrics for intent-classification. If intent not found, assume False
        - Acc. Detection of slots
        - Metrics for slot-classification. If intent not found, assume False
    """

    def __init__(
        self,
        num_intents: int,
        num_slots: int,
    ):
        super(IntentSlotMatch, self).__init__()

        # Init states to store results of exact-ontology-match, intent-classification, and slot-classification
        for token_type in ONTOLOGY_TYPE_LIST:
            self.add_state(
                "precision_{}".format(token_type),
                default=torch.tensor(0.0),
                dist_reduce_fx="sum",
            )
            self.add_state(
                "recall_{}".format(token_type),
                default=torch.tensor(0.0),
                dist_reduce_fx="sum",
            )
            self.add_state(
                "f1_{}".format(token_type),
                default=torch.tensor(0.0),
                dist_reduce_fx="sum",
            )

        # Initialize metric functions
        self.precision_map = {
            "ontology": Precision(
                task="binary", num_classes=2, ignore_index=IGNORED_INDEX
            ),
            "intents": Precision(
                task="multiclass", num_classes=num_intents, ignore_index=IGNORED_INDEX
            ),
            "slots": Precision(
                task="multiclass", num_classes=num_slots, ignore_index=IGNORED_INDEX
            ),
        }
        self.recall_map = {
            "ontology": Recall(
                task="binary", num_classes=2, ignore_index=IGNORED_INDEX
            ),
            "intents": Recall(
                task="multiclass", num_classes=num_intents, ignore_index=IGNORED_INDEX
            ),
            "slots": Recall(
                task="multiclass", num_classes=num_slots, ignore_index=IGNORED_INDEX
            ),
        }
        self.f1score_map = {
            "ontology": F1Score(
                task="binary", num_classes=2, ignore_index=IGNORED_INDEX
            ),
            "intents": F1Score(
                task="multiclass", num_classes=num_intents, ignore_index=IGNORED_INDEX
            ),
            "slots": F1Score(
                task="multiclass", num_classes=num_slots, ignore_index=IGNORED_INDEX
            ),
        }

    def update(
        self,
        batch: ParseOutputs,
        token_type: str,
    ) -> None:
        assert token_type in ONTOLOGY_TYPE_LIST, ValueError(
            "{} is not a valid token_type".format(token_type)
        )

        # Compute metrics: exact-ontology-match, intent-classification, and slot-classification
        if token_type == "ontology":
            self.precision_ontology += self.precision_map[token_type](
                batch.outputs, batch.targets
            )
            self.recall_ontology += self.recall_map[token_type](
                batch.outputs, batch.targets
            )
            self.f1_ontology += self.f1score_map[token_type](
                batch.outputs, batch.targets
            )
        elif token_type == "intents":
            self.precision_intents += self.precision_map[token_type](
                batch.outputs, batch.targets
            )
            self.recall_intents += self.recall_map[token_type](
                batch.outputs, batch.targets
            )
            self.f1_intents += self.f1score_map[token_type](
                batch.outputs, batch.targets
            )
        elif token_type == "slots":
            self.precision_slots += self.precision_map[token_type](
                batch.outputs, batch.targets
            )
            self.recall_slots += self.recall_map[token_type](
                batch.outputs, batch.targets
            )
            self.f1_slots += self.f1score_map[token_type](batch.outputs, batch.targets)

    def compute(self) -> None:
        return {
            "precision_ontology": self.precision_ontology,
            "precision_intents": self.precision_intents,
            "precision_slots": self.precision_slots,
            "recall_ontology": self.recall_ontology,
            "recall_intents": self.recall_intents,
            "recall_slots": self.recall_slots,
            "f1_ontology": self.f1_ontology,
            "f1_intents": self.f1_intents,
            "f1_slots": self.f1_slots,
        }


# TO-DO: F1 and Acc. for all intents and slots using frequency

# TO-DO: labeled bracketing for intents, and intent-slots

# Tree validity - via bracket matching

# BLEU score
