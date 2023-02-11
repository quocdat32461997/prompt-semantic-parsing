import torch
from typing import List
from torch import Tensor
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from psp.transforms import InputTransform
from psp.constants import (
    ListInputs,
    ParseInputs,
    RunMode,
    DatasetPaths,
    PRETRAINED_BART_MODEL
)
from psp.dataset.tokenizer import Tokenizer, PointerTokenizer
from psp.dataset.datasets import LowResourceTOPDataset, LowResourceTOPv2Dataset
from psp.transforms.input_transform import TokenTransform, InputTransform, TextTransform
class SemanticParseDataModule(pl.LightningDataModule):
    "Casual Semantic-Parse Data Module"
    def __init__(
        self,
        dataset_name: str,
        batch_size: int,
        num_workers: int = 2, 
        pretrained: str = None, # default, PRETRAINED_BART_MODEL
        use_pointer_data: bool = False,
        use_processed_data: bool = False,
        **kwargs
    ):
        super(SemanticParseDataModule, self).__init__()
        if use_pointer_data and not use_processed_data:
            raise ValueError("use_pointer_data and use_processed_data must be both True")
        if dataset_name == "topv2":
            dataset_path = DatasetPaths.TOPv2
            self.dataset = LowResourceTOPv2Dataset
        elif dataset_name == "top":
            dataset_path = DatasetPaths.TOP
            self.dataset = LowResourceTOPDataset
        else:
            raise ValueError("{} dataset is not a valid choie.".format(dataset_name))
        if not pretrained:
            pretrained = PRETRAINED_BART_MODEL
    
        # Init transform
        self.transform: InputTransform = None
        if use_processed_data:
            self.transform = TokenTransform(
                pretrained=pretrained,
                dataset_path=dataset_path,
                use_pointer_data=use_pointer_data
            )
        else:
            self.transform = TextTransform(
                pretrained=pretrained,
                dataset_path=dataset_path
            )
        self.batch_size: int = batch_size
        self.num_workers: int = num_workers
    def setup(self, stage: str):
        if stage == "fit":
            self.train_set = self.dataset(bucket=RunMode.TRAIN)
            self.val_set = self.dataset(bucket=RunMode.EVAL)
        
        if stage == "test":
            self.test_set = self.dataset(bucket=RunMode.TEST)
    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=self.num_workers)
    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.num_workers)