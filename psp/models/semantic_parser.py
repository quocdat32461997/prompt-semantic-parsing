import torch
from typing import Dict, List
import pytorch_lightning as pl
from torch.nn import Module
from torch import Tensor
import torch.nn.functional as F
from torch.optim import Adam

from psp.constants import ParseInputs, RunMode, IGNORED_INDEX, ONTOLOGY_TYPE_LIST
from psp.models.optimizers import MAMLOptimizer
from psp.models.metrics import ExactMatch, IntentSlotMatch
from psp.models.model_utils import pad_tensors
from psp.dataset.transforms import ParseTransform


class SemmanticParser(pl.LightningModule):
    """Uses BART as the core model."""

    def __init__(
        self,
        model: Module,
        lr: float,
        intent_id_list: List[int],
        slot_id_list: List[int],
        ontology_id_list: List[int],
        vocab_size: int,
    ):
        super().__init__()
        self.model: Module = model
        self.lr: int = lr

        device = torch.cuda.current_device()

        # build metrics and parse_transform
        self.parse_transform: ParseTransform = ParseTransform(
            intent_id_list, slot_id_list, ontology_id_list, vocab_size, device=device)
        self.build_metrics(num_intents=len(intent_id_list), num_slots=len(slot_id_list), device=device)

        # save hyperparameters
        self.save_hyperparameters(ignore=["model"])

    def build_metrics(self, num_intents: int, num_slots: int, device: str='cpu') -> None:
        # Exact Match (EM) Acc.
        self.em_acc = ExactMatch().to(device)

        # Precision, Recall, and F1Score for detecting intents and slots
        self.intent_slot_metrics = IntentSlotMatch(
            num_intents=num_intents, num_slots=num_slots
        ).to(device)

    def _ignore_tokens(self, tensors: Tensor):
        """
        Ignores <BOS> and <PAD> for fair evaluation"""
        return torch.where(
            (tensors != self.model.pad_token_id) | (tensors != self.model.bos_token_id),
            tensors,
            torch.full(tensors.shape, IGNORED_INDEX),
        )


class LowResourceSemanticParser(SemmanticParser):
    def __init__(
        self,
        model: Module,
        lr: float,
        intent_id_list: List[int],
        slot_id_list: List[int],
        ontology_id_list: List[int],
        vocab_size: int,
        label_smoothing: float = 0.1,
    ) -> None:
        super(LowResourceSemanticParser, self).__init__(
            model=model,
            lr=lr,
            intent_id_list=intent_id_list,
            slot_id_list=slot_id_list,
            ontology_id_list=ontology_id_list,
            vocab_size=vocab_size,
        )

        self.loss_fn = torch.nn.CrossEntropyLoss(
            ignore_index=self.model.pad_token_id,
            label_smoothing=label_smoothing)

    def compute_loss(self, outputs: Tensor, batch: ParseInputs) -> Tensor:
        """
        Args:
            outputs: Tensor outputs in shape [batch_size, seq_len - 1, vocab_size] that without <BOS> at the beggining
            batch: ParseInputs object to store semantic_parse_ids (aka gold references)
                semantic_parse_ids: [batch_size, seq_len]
        """
        # Skip <BOS> in references
        semantic_parse: Tensor = batch.semantic_parse_ids[:, 1:]

        # Outputs and gold references must match the sequence length
        assert outputs.shape[1] == semantic_parse.shape[-1]

        # reshape outputs to [batch_size, vocab_size, seq_len]
        outputs = torch.reshape(
            outputs, (outputs.shape[0], outputs.shape[2], outputs.shape[1])
        )
        return self.loss_fn(outputs, semantic_parse)

    def compute_metrics(self, outputs: Tensor, batch: ParseInputs) -> Dict[str, Tensor]:
        """
        Args:
            - outputs: Tensor of shape [batch_size, seq_len_A]
            - batch: ParseInputs object to store gold references (semantic_parse_ids)
                semantic_parse_ids: Tensor of shape [batch_size, seq_len_B]
            **NOTE**: seq_len_A may not equal to seq_len_B
        """
        # Reterieve outputs and gold references
        targets: Tensor = batch.semantic_parse_ids

        # Padding if unequal length
        max_length: int = max(targets.shape[-1], outputs.shape[-1])
        outputs = pad_tensors(
            outputs, len(outputs), max_length, value=self.model.pad_token_id
        )
        targets = pad_tensors(
            targets, len(outputs), max_length, value=self.model.pad_token_id
        )

        metrics: Dict[str, Tensor] = {}
        # Exact Match Acc.
        metrics.update(self.em_acc(outputs, targets))

        # Ontology-levl, Intent-level and Slot-level metrics
        for token_type in ONTOLOGY_TYPE_LIST:
            metrics.update(
                self.intent_slot_metrics(
                    batch=self.parse_transform(
                        outputs=outputs, targets=targets, token_type=token_type
                    ),
                    token_type=token_type,
                ),
            )

        # BLEU score
        return metrics

    def configure_optimizers(self):
        optimizer = Adam(
            self.model.parameters(), lr=self.lr
        )  # MAMLOptimizer(self.model.parameters(), self.lr)

        return optimizer

    def _run(self, batch: ParseInputs, run_mode: RunMode) -> Tensor:

        # Forward
        outputs: Tensor = self.model(batch, run_mode=run_mode)

        metrics = {}
        if run_mode == RunMode.TRAIN:
            # Compute loss for training only
            loss = self.compute_loss(outputs, batch)
            metrics["loss"] = loss
        else:
            # Compute metrics
            metrics.update(self.compute_metrics(outputs, batch))

        # Log metrics
        for name, value in metrics.items():
            self.log(
                name, value, on_step=True, on_epoch=True, prog_bar=True, logger=True
            )

        return metrics

    def training_step(self, batch: ParseInputs, batch_idx: int):
        return self._run(batch, run_mode=RunMode.TRAIN)

    def validattion_step(self, batch: ParseInputs, batch_idx: int):
        return self._run(batch, run_mode=RunMode.EVAL)

    def test_step(self, batch: ParseInputs, batch_idx: int):
        return self._run(batch, run_mode=RunMode.TEST)
