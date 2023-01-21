import torch
from torchmetrics import Metric, F1Score, Precision, Recall
from torch import Tensor
from psp.constants import IGNORED_INDEX


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
    ):
        super(IntentSlotMatch, self).__init__()

        # Detecting ontology tokens
        self.add_state(
            "precision_ontology_tokens", default=torch.tensor(0), dist_reduce_fx="sum"
        )
        self.add_state(
            "recall_ontology_tokens", default=torch.tensor(0), dist_reduce_fx="sum"
        )
        self.add_state(
            "f1_ontology_tokens", default=torch.tensor(0), dist_reduce_fx="sum"
        )

        # Detecting and classifying intents
        self.add_state(
            "precision_intents", default=torch.tensor(0), dist_reduce_fx="sum"
        )
        self.add_state("recall_intents", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("f1_intents", default=torch.tensor(0), dist_reduce_fx="sum")

        # Detecting and classifying slots
        self.add_state("precision_slots", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("recall_slots", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("f1_slots", default=torch.tensor(0), dist_reduce_fx="sum")

        # Initialize metric functions
        self._build_metrics()

    def compute_metrics(self, outputs: Tensor, targets: Tensor, type_name: str) -> None:
        if type_name == "ontology_tokens":
            self.precision_ontology_tokens += self.precision(outputs, targets)
            self.recall_ontology_tokens += self.recall(outputs, targets)
            self.f1_ontology_tokens += self.f1score(outputs, targets)
        elif type_name == "intents":
            self.precision_intents += self.precision(outputs, targets)
            self.recall_intents += self.recall(outputs, targets)
            self.f1_intents += self.f1score(outputs, targets)
        elif type_name == "slots":
            self.precision_slots += self.precision(outputs, targets)
            self.recall_slots += self.recall(outputs, targets)
            self.f1_slots += self.f1score(outputs, targets)
        else:
            raise ValueError("{} is not a valid type_name".format(type_name))

    def _build_metrics(self) -> None:
        self.precision = Precision(
            task="binary", num_classes=2, ignore_index=IGNORED_INDEX
        )
        self.recall = Recall(task="binary", num_classes=2, ignore_index=IGNORED_INDEX)
        self.f1score = F1Score(task="binary", num_classes=2, ignore_index=IGNORED_INDEX)

    def update(
        self,
        preds: Tensor,
        targets: Tensor,
        ontology_token_mask: Tensor,
        intent_mask: Tensor,
        slot_mask: Tensor,
    ) -> None:
        assert preds.shape == targets.shape

        # update total slots, intents, and ontology tokens
        # self.total_ontology_tokens += batch.ontology_token_mask.sum()
        # self.total_intents += batch.intent_mask.sum()
        # self.total_slots += batch.slot_mask.sum()

        preds = torch.mul(preds, ontology_token_mask)

        # Metrics for ontology tokens
        targets = torch.mul(targets, ontology_token_mask)
        mask = torch.add(ontology_token_mask * -IGNORED_INDEX, IGNORED_INDEX)
        targets = torch.add(targets, mask)
        self.compute_metrics(preds, targets, type_name="ontology_tokens")

        # Metrics for intents
        intent_targets = torch.mul(targets, intent_mask)
        mask = torch.add(intent_mask * -IGNORED_INDEX, IGNORED_INDEX)
        intent_targets = torch.add(intent_targets, mask)
        self.compute_metrics(preds, intent_targets, type_name="intents")

        # Metrics for slots
        slot_targets = torch.mul(targets, slot_mask)
        mask = torch.add(slot_mask * -IGNORED_INDEX, IGNORED_INDEX)
        slot_targets = torch.add(slot_targets, mask)
        self.compute_metrics(preds, slot_targets, type_name="slots")

    def compute(self) -> None:
        return {
            "precision-ontology-tokens": self.precision_ontology_tokens,
            "precision-intents": self.precision_intents,
            "precision-slots": self.precision_slots,
            "recall-ontology-tokens": self.recall_ontology_tokens,
            "recall-intents": self.recall_intents,
            "recall-slots": self.recall_slots,
            "f1-ontology-tokens": self.f1_ontology_tokens,
            "f1-intents": self.f1_intents,
            "f1-slots": self.f1_slots,
        }


# TO-DO: F1 and Acc. for all intents and slots using frequency

# TO-DO: labeled bracketing for intents, and intent-slots

# Tree validity - via bracket matching

# BLEU score

# Slot-Detection: compare indices of slots (including intents)
# Slot-Classification: if found slot, correct slot?
# Slot-span accuracy: within-span only

# Intent-Detection
# Intent-Classifcation
# Intent-span accuracy
