from typing import List
import pandas as pd
import torch
from torch import Tensor

from psp.constants import ParseInputs, ListInputs


def read_topv2_dataset(paths: List[str]) -> pd.DataFrame:
    # Read and merge data (dedicated to TOPV2 dataset)
    dfs = [pd.read_csv(path, sep="\t") for path in paths]
    dfs = pd.concat(dfs)
    return dfs


def read_top_dataset(path: str) -> pd.DataFrame:
    # Read data
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
    )
    # Rename column for easy accees
    df = df.rename(columns={0: "utterance", 2: "semantic_parse"})
    return df
