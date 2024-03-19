import os
import pickle
import pandas as pd
from typing import List, Dict, Union
from torch import Tensor
import torch
from torch.utils.data import Dataset

from psp.constants import (
    ListInputs,
    TOPv2Domain,
    TOPDomain,
    DatasetPaths,
    RunMode,
)
from psp.dataset.data_utils import read_top_dataset, read_topv2_dataset


class TOPv2Dataset(Dataset):
    """TOPV2 dataset"""

    #BUCKET_DICT: Dict[str, str] = {
    #    "train": "_train.tsv",
    #    "eval": "_eval.tsv",
    #    "test": "_test.tsv",
    #}

    def __init__(self, bucket: RunMode, use_processed_data: bool = False, **kwargs) -> None:
        super().__init__()

        self.use_processed_data: bool = use_processed_data
        
        # Read data
        self.data: Union[pd.DataFrame, Dict[str, Tensor]] = None
        
        if not self.use_processed_data:
            self.data = read_topv2_dataset(
            [
                os.path.join(
                    DatasetPaths.TOPv2.value,
                    domain.name + "_{}.tsv" + bucket.value,
                )
                for domain in TOPv2Domain
            ]
        )
        else:
            with open(
                os.path.join(DatasetPaths.TOPv2.value, "processed_{}.pkl".format(bucket.value)), 'rb') as file:
                self.data = pickle.load(file)
            #self.data = pd.read_csv(os.path.join(DatasetPaths.TOPv2.value, "processed_{}.tsv".format(bucket.value)))

    def _cast(self, inputs: Union[str, List[int]]) -> Union[str, Tensor]:
        """
        If self.use_processed_data=True, expected inputs of type List[int].
        Otherwise, expected inputs of str type.
        """
        return torch.tensor(inputs) if self.use_processed_data else inputs
    
    def __len__(self) -> int:
        return len(self.data)


class TOPDataset(Dataset):
    """
    TOP dataset
    """

    BUCKET_DICT: Dict[str, str] = {
        "train": "train.tsv",
        "eval": "eval.tsv",
        "test": "test.tsv",
    }

    def __init__(self, bucket: RunMode, **kwargs) -> None:
        super().__init__()

        # Read data
        self.data: pd.DataFrame = read_top_dataset(
            os.path.join(DatasetPaths.TOP.value, TOPDataset.BUCKET_DICT[bucket.value])
        )

    def __len__(self) -> int:
        return len(self.data)


class LowResourceTOPDataset(TOPDataset):
    def __getitem__(self, idx) -> ListInputs:
        sample = self.data.iloc[idx] if isinstance(self.data, pd.DataFrame) else self.data[idx]
        return ListInputs(
            domain=TOPDomain.none,
            utterance=sample["utterance"] ,
            semantic_parse=sample["semantic_parse"],
            pointer_parse=sample["pointer_parse"] if "pointer_parse" in sample else None,
        )


class LowResourceTOPv2Dataset(TOPv2Dataset):
    def __getitem__(self, idx) -> ListInputs:
        sample = self.data.iloc[idx] if isinstance(self.data, pd.DataFrame) else self.data[idx]

        return ListInputs(
            domain=TOPv2Domain[sample["domain"]].value,
            utterance=self._cast(sample["utterance"]),
            semantic_parse=self._cast(sample["semantic_parse"]),
            pointer_parse=self._cast(sample["pointer_parse"]) if "pointer_parse" in sample else None,
        )

class PromptTOPv2Dataset(TOPv2Dataset):
    def __getitem__(self, idx) -> ListInputs:
        return None
