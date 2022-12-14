from typing import List
import pandas as pd
import torch
from torch import Tensor

from psp.constants import ParseInputs, ListInputs


def read_and_merge(paths: List[str]) -> pd.DataFrame:
    # Read and merge data
    dfs = [pd.read_csv(path, sep='\t') for path in paths]
    dfs = pd.concat(dfs)
    return dfs
