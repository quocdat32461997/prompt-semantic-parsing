import torch
from typing import Dict
import pytorch_lightning as pl
from torch.nn import Module
from torch import Tensor
import torch.nn.functional as F
from torch.optim import Adam

from psp.constants import ParseInputs, RunMode, IGNORED_INDEX
from psp.models.optimizers import MAMLOptimizer
from psp.models.metrics import ExactMatch, IntentSlotMatch


def pad_tensors(
    tensors: Tensor, batch_size: int, max_length: int, value: int = IGNORED_INDEX
):
    """
    Pad 2D tensors
    """
    assert len(tensors.shape) == 2
    return torch.cat(
        [
            tensors,
            torch.full((batch_size, max_length - tensors.shape[-1]), value),
        ],
        dim=-1,
    )


class SemmanticParser(pl.LightningModule):
    """Uses BART as the core model."""

    def __init__(self, model: Module, lr: float):
        super().__init__()
        self.model: Module = model
        self.lr: int = lr

        # build metrics
        self.build_metrics()

        # save hyperparameters
        self.save_hyperparameters(ignore=["model"])

    def build_metrics(self) -> None:
        # Exact Match (EM) Acc.
        self.em_acc = ExactMatch()

        # Precision, Recall, and F1Score for detecting intents and slots
        self.intent_slot_metrics = IntentSlotMatch()

    def _ignore_tokens(self, tensors: Tensor):
        """
        Ignores <BOS> and <PAD> for fair evaluation"""
        return torch.where(
            (tensors != self.model.pad_token_id) | (tensors != self.model.bos_token_id),
            tensors,
            torch.full(tensors.shape, IGNORED_INDEX),
        )


class LowResourceSemanticParser(SemmanticParser):
    def __init__(self, model: Module, lr: float) -> None:
        super(LowResourceSemanticParser, self).__init__(model=model, lr=lr)

        self.loss_fn = torch.nn.NLLLoss()

    def compute_loss(self, outputs: Tensor, batch: ParseInputs) -> Tensor:
        """
        Args:
            outputs: Tensor outputs in shape [batch_size, seq_len, vocab_size]
            batch: ParseInputs object to store semantic_parse_ids (aka gold references)
                semantic_parse_ids: [batch_size, seq_len]
        """
        # Outputs and gold references must match the sequence length
        assert outputs.shape[1] == batch.semantic_parse_ids.shape[-1]

        # Ignore <PAD> and <BOS>
        semantic_parse: Tensor = self._ignore_tokens(batch.semantic_parse_ids)
        outputs = self._ignore_tokens(outputs)

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

        # Ignore <BOS> and <PAD> tokens
        # outputs = self._ignore_tokens(outputs)
        # targets = self._ignore_tokens(targets)

        ontology_token_mask: Tensor = pad_tensors(
            batch.ontology_token_mask, len(outputs), max_length
        )
        intent_mask: Tensor = pad_tensors(batch.intent_mask, len(outputs), max_length)
        slot_mask: Tensor = pad_tensors(batch.slot_mask, len(outputs), max_length)

        metrics: Dict[str, Tensor] = {}
        # Exact Match Acc.
        metrics.update(self.em_acc(outputs, targets))

        # Intent-level and Slot-level metrics
        metrics.update(
            self.intent_slot_metrics(
                outputs, targets, ontology_token_mask, intent_mask, slot_mask
            )
        )

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
