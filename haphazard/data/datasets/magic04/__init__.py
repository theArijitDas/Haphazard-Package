"""
haphazard.data.datasets.magic04
-------------------------------
This module implements a magic04 dataset
"""

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ...base_dataset import BaseDataset
from ...datasets import register_dataset
from ....utils.file_utils import find_file


@register_dataset("magic04")
class Magic04(BaseDataset):
    """
    Magic04 Dataset
    """

    def __init__(self, base_path: str = "./", **kwargs) -> None:
        """
        Magic04 dataset for binary classification.

        Args:
            base_path (str): Base path (default "./").
        
        Notes:
        - The raw dataset file has ordered labels (first 12332 '1', rest '0').
        - Data is shuffled to avoid bias during online training.
        """
        self.name = "magic04"
        self.haphazard_type = "controlled"
        self.task = "classification"
        self.num_classes = 2

        super().__init__(base_path=base_path, **kwargs)
    
    def read_data(self, base_path: str = ".") -> tuple[NDArray[np.float64], NDArray[np.int64]]:
        """
        Load the magic04 dataset.

        Args:
            base_path (str): Directory under which the dataset file is searched.

        Returns:
            Tuple[NDArray[np.float64], NDArray[np.int64]]:
                - x: Feature matrix of shape (n_samples, n_features) with dtype float64.
                - y: Binary label vector of shape (n_samples,) with dtype int64.
        """
        data_path: str = find_file(base_path, "magic04.data")
        df: pd.DataFrame = pd.read_csv(data_path, sep=",", header=None, engine="python")

        # Shuffle rows to randomize label order
        df = df.sample(frac=1.0, random_state=42)

        # Features and labels
        x = df.iloc[:, :10].to_numpy()
        y = (df[10] == 'g').astype(int).to_numpy()
        return x, y

__all__ = [
    "Magic04"
]
