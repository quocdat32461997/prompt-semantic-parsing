import torch
from typing import Dict
import pytorch_lightning as pl
from torch.nn import Module
from torch import Tensor
import torch.nn.functional as F
from torch.optim import Adam

from psp.constants import ParseInputs, RunMode
from psp.models.optimizers import MAMLOptimizer


class SemmanticParser(pl.LightningModule):
    """Uses BART as the core model."""

    def __init__(self, model: Module, lr: float):
        super().__init__()
        self.model: Module = model
        self.lr: int = lr

        # save hyperparameters
        self.save_hyperparameters(ignore=["model"])

    def build_metrics(self) -> None:
        raise NotImplementedError


class LowResourceSemanticParser(SemmanticParser):
    def __init__(self, model: Module, lr: float):
        super(LowResourceSemanticParser, self).__init__(model=model, lr=lr)

        self.loss_fn = torch.nn.NLLLoss()

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
            batch.semantic_parse_ids,
            torch.full(batch.semantic_parse_ids.shape, -100),
        )
        semantic_parse = semantic_parse[:, 1:]  # Ignore BOS tokens

        # reshape outputs to [batch_size, vocab_size, seq_len]
        outputs = torch.reshape(
            outputs, (outputs.shape[0], outputs.shape[2], outputs.shape[1])
        )
        return self.loss_fn(outputs, semantic_parse)  # ignore_index = -100

    def compute_metrics(self, outputs: Tensor, batch: ParseInputs) -> Dict[str, Tensor]:
        # Exact Match Acc.
        self.em_acc = None

        # Intent metrics
        self.intent_metrics = None
        self.slot_metrics = None

        return None

    def configure_optimizers(self):
        optimizer = Adam(
            self.model.parameters(), lr=self.lr
        )  # MAMLOptimizer(self.model.parameters(), self.lr)

        return optimizer

    def _run(self, batch: ParseInputs, run_mode: RunMode) -> Tensor:

        # Forward
        outputs: Tensor = self.model(batch)

        # Compute loss
        loss = self.compute_loss(outputs, batch)
        metrics = {"loss": loss}

        if run_mode == RunMode.EVAL or run_mode == RunMode.TEST:
            # Compute and log metrics
            outputs = self.model.predict(batch)
            metrics.update(self.compute_metrics(outputs, batch))

        # Log metrics
        for name, value in metrics.items():
            self.log(
                name, value, on_step=True, on_epoch=True, prog_bar=True, logger=True
            )

        return metrics

    def training_step(self, batch: ParseInputs):
        return self._run(batch, run_mode=RunMode.TRAIN)

    def validattion_step(self, batch: ParseInputs):
        return self._run(batch, run_mode=RunMode.EVAL)

    def test_step(self, batch: ParseInputs):
        return self._run(batch, run_mode=RunMode.TEST)
