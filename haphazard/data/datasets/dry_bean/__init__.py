"""
haphazard.data.datasets.dry_bean
--------------------------------
This module implements a dry_bean dataset
"""

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.io import arff

from ...base_dataset import BaseDataset
from ...datasets import register_dataset
from ....utils.file_utils import find_file


@register_dataset("dry_bean")
class DryBean(BaseDataset):
    """
    DryBean Dataset
    """

    def __init__(self, base_path: str = "./", **kwargs) -> None:
        """
        DryBean dataset for multi-class classification.

        Args:
            base_path (str): Base path (default "./").
        
        Notes:
        - Class labels are mapped from strings to integers.
        - Original dataset has ordered labels as follows:
            Index         -> Label
            0 to 2026     -> 0
            2027 to 3348  -> 1
            3349 to 3870  -> 2
            3871 to 5500  -> 3
            5501 to 7428  -> 4
            7429 to 10064 -> 5
            10065 to 13610-> 6
        - Data is shuffled for randomized experiments.
        """
        self.name = "dry_bean"
        self.haphazard_type = "controlled"
        self.task = "classification"
        self.num_classes = 7

        super().__init__(base_path=base_path, **kwargs)
    
    def read_data(self, base_path: str = ".") -> tuple[NDArray[np.float64], NDArray[np.int64]]:
        """
        Load the dry_bean dataset.

        Args:
            base_path (str): Directory under which the dataset file is searched.

        Returns:
            Tuple[NDArray[np.float64], NDArray[np.int64]]:
                - x: Feature matrix of shape (n_samples, n_features) with dtype float64.
                - y: Binary label vector of shape (n_samples,) with dtype int64.
        """
        data_path: str = find_file(base_path, "Dry_Bean_Dataset.arff")

        data, meta = arff.loadarff(data_path)
        df: pd.DataFrame = pd.DataFrame(data)

        # Decode string labels and map to integers
        df["Class"] = df["Class"].str.decode("utf-8")
        class_encoding: dict[str, int] = {
            "SEKER": 0,
            "BARBUNYA": 1,
            "BOMBAY": 2,
            "CALI": 3,
            "HOROZ": 4,
            "SIRA": 5,
            "DERMASON": 6,
        }
        df["Class"] = df["Class"].map(class_encoding)

        # Shuffle rows to randomize label order
        df = df.sample(frac=1.0, random_state=42)

        x = df.drop(columns=["Class"]).to_numpy(dtype=np.float64)
        y = df["Class"].to_numpy(dtype=np.int64)

        return x, y

__all__ = [
    "DryBean"
]
