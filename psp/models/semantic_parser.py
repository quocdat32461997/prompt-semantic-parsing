import pytorch_lightning as pl

from transformers import BartModel

class SemmanticParser(pl.LightningDataModule):
    """Uses BART as the core model.
    """
    def __init__(self, pretrained: str):
        super().__init__()
        self.model = BartModel.from_pretrained(pretrained)

        # save hyperparameters
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError

    def test_step(self, batch, batch_idx):
        raise NotImplementedError