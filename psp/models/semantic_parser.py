import torch
from typing import Dict
import pytorch_lightning as pl
from torch.nn import Module
from torch import Tensor
import torch.nn.functional as F

from psp.constants import ParseInputs
from psp.models.optimizers import MAMLOptimizer
from psp.models import Seq2SeqCopyPointer


class SemmanticParser(pl.LightningDataModule):
    """Uses BART as the core model."""

    def __init__(self, model: Module, lr: float):
        super().__init__()
        self.model: Module = model
        self.lr: int = lr

        # save hyperparameters
        self.save_hyperparameters()

    def build_metrics(self) -> None:
        raise NotImplementedError


class LowResourceSemanticParser(SemmanticParser):
    def __init__(self, model: Module):
        super(LowResourceSemanticParser, self).__init__(model=model)

    def build_metrics(self) -> None:
        # Exact Match (EM) Acc.
        self.em_acc = None

        # Intent Acc. and F1.
        self.intent_metrics = None

        # Slot Acc. and F1.
        self.slot_metrics = None

        pass

    def compute_loss(self, outputs: Tensor, batch: ParseInputs) -> Tensor:
        # Mask PAD_TOKENS
        semantic_parse: Tensor = torch.where(
            batch.semantic_parse_attn_mask != 0,
            batch.semantic_parse,
            torch.full(batch.semantic_parse.shape, -100),
        )
        return F.nll_loss(outputs, semantic_parse)  # ignore_index = -100

    def compute_metrics(self, outputs: Tensor, batch: ParseInputs) -> Dict[str, Tensor]:
        # Exact Match Acc.
        self.em_acc = None

        # Intent metrics
        self.intent_metrics = None
        self.slot_metrics = None

        return None

    def configure_optimizer(self):
        self.optimizer = MAMLOptimizer(self.model.parameters(), self.lr)

    def _run(self, batch: ParseInputs) -> Tensor:
        # Forward
        outputs: Tensor = self.model(batch)

        # Compute loss
        loss = self.compute_loss(outputs, batch)

        # Predict and compute metrics
        outputs = self.model.predict(batch)
        metrics = self.compute_metrics(outputs, batch)

        # Add loss to metrics-dict
        metrics["loss"] = loss

        for name, value in metrics.items():
            self.log(
                name, value, on_step=True, on_epoch=True, prog_bar=True, logger=True
            )

        return metrics

    def training_step(self, batch: ParseInputs):
        return self._run(batch)

    def validattion_step(self, batch: ParseInputs):
        return self._run(batch)

    def test_step(self, batch: ParseInputs):
        return self._run(batch)
