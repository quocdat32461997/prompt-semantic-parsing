import os
import pandas as pd
from typing import List, Dict
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
    """
    TOPV2 dataset"""

    BUCKET_DICT: Dict[str, str] = {
        "train": "_train.tsv",
        "eval": "_eval.tsv",
        "test": "_test.tsv",
    }

    def __init__(self, bucket: RunMode) -> None:
        super().__init__()

        # Read data
        self.data: pd.DataFrame = read_topv2_dataset(
            [
                os.path.join(
                    DatasetPaths.TOPv2.value,
                    domain.name + TOPv2Dataset.BUCKET_DICT[bucket.value],
                )
                for domain in TOPv2Domain
            ]
        )

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

    def __init__(self, bucket: RunMode) -> None:
        super().__init__()

        # Read data
        self.data: pd.DataFrame = read_top_dataset(
            os.path.join(DatasetPaths.TOP.value, TOPDataset.BUCKET_DICT[bucket.value])
        )

    def __len__(self) -> int:
        return len(self.data)


class LowResourceTOPDataset(TOPDataset):
    def __getitem__(self, idx) -> ListInputs:
        sample = self.data.iloc[idx]
        return ListInputs(
            domain=TOPDomain.none,
            utterance=sample["utterance"],
            semantic_parse=sample["semantic_parse"],
            pointer_parse=sample["pointer_parse"] if "pointer_parse" in sample.columns else None,
        )


class LowResourceTOPv2Dataset(TOPv2Dataset):
    def __getitem__(self, idx) -> ListInputs:
        sample = self.data.iloc[idx]

        # Encode domain
        domain: int = TOPv2Domain[sample["domain"]].value

        return ListInputs(
            domain=domain,
            utterance=sample["utterance"],
            semantic_parse=sample["semantic_parse"],
            pointer_parse=sample["pointer_parse"] if "pointer_parse" in sample.columns else None,
        )

class PromptTOPv2Dataset(TOPv2Dataset):
    def __getitem__(self, idx) -> ListInputs:
        return None
