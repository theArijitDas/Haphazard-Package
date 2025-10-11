"""
haphazard.data.datasets.gas
--------------------------------
This module implements a gas dataset
"""

import numpy as np
from numpy.typing import NDArray

from ...base_dataset import BaseDataset
from ...datasets import register_dataset
from ....utils.file_utils import find_file


@register_dataset("gas")
class Gas(BaseDataset):
    """
    Gas Dataset
    """

    def __init__(self, base_path: str = "./", **kwargs) -> None:
        """
        Gas dataset for multi-class classification.

        Args:
            base_path (str): Base path (default "./").
        
        Notes:
        - Each batch file contains lines in the format:
            label;... feature_index:feature_value ...
        - Labels are adjusted to zero-based indexing.
        - Raises ValueError if any feature vector is not length 128.
        """
        self.name = "gas"
        self.haphazard_type = "controlled"
        self.task = "classification"
        self.num_classes = 6

        super().__init__(base_path=base_path, **kwargs)
    
    def read_data(self, base_path: str = ".") -> tuple[NDArray[np.float64], NDArray[np.int64]]:
        """
        Load the gas dataset.

        Args:
            base_path (str): Directory under which the dataset file is searched.

        Returns:
            Tuple[NDArray[np.float64], NDArray[np.int64]]:
                - x: Feature matrix of shape (n_samples, n_features) with dtype float64.
                - y: Binary label vector of shape (n_samples,) with dtype int64.
        """
        x_list: list[list[float]] = []
        y_list: list[int] = []
        for i in range(1, 11):
            file_path = find_file(base_path, f"gas/batch{i}.dat")
            with open(file_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    label = int(parts[0].split(';')[0]) - 1  # zero-based
                    y_list.append(label)
                    features = [float(p.split(':')[1]) for p in parts[1:]]
                    if len(features) != 128:
                        raise ValueError(f"Feature length mismatch in {file_path}: {len(features)}")
                    x_list.append(features)

        x = np.array(x_list, dtype=np.float64)
        y = np.array(y_list, dtype=np.int64)
        return x, y

__all__ = [
    "Gas"
]
