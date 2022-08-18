import pytorch_lightning as pl
from torch.nn import Module

from transformers import BartModel

from psp.constants import ParseInputs
from psp.models.optimizers import MAMLOptimizer


class SemmanticParser(pl.LightningDataModule):
    """Uses BART as the core model.
    """

    def __init__(self, model: Module, lr: float):
        super().__init__()
        self.model: Module = model
        self.lr: int = lr

        # save hyperparameters
        self.save_hyperparameters()

    def build_metrics(self) -> None:
        raise NotImplementedError

    def setup(self) -> None:
        raise NotImplementedError


class LowResourceSemanticParser(SemmanticParser):
    def training_step(self, batch: ParseInputs):
        pass
    def configure_optimizer(self):
        self.optimizer = MAMLOptimizer(self.model.parameters(), self.lr)



class DiscretePromptSemanticParser(SemmanticParser):
    def __init__(self, model: Module):
        super().__init__(model=model)
