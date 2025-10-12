"""
haphazard.data.datasets.gas
---------------------------
Gas dataset implementation for multi-class classification.
"""

import numpy as np
from numpy.typing import NDArray

from ...base_dataset import BaseDataset
from ...datasets import register_dataset
from ....utils.file_utils import find_file


@register_dataset("gas")
class Gas(BaseDataset):
    """
    Gas dataset for multi-class classification.

    Attributes
    ----------
    name : str
        Name of the dataset.
    haphazard_type : str
        Indicates dataset type, e.g., "controlled".
    task : str
        Task type, e.g., "classification".
    num_classes : int
        Number of unique output classes.

    Notes
    -----
    - Each batch file contains lines in the format:
        label; ... feature_index:feature_value ...
    - Labels are adjusted to zero-based indexing.
    - Raises `ValueError` if any feature vector is not of length 128.
    """

    def __init__(self, base_path: str = "./", **kwargs) -> None:
        """
        Initialize the Gas dataset instance.

        Parameters
        ----------
        base_path : str, default="./"
            Base directory under which the dataset files are located.
        **kwargs
            Additional keyword arguments passed to the parent constructor.
        """
        self.name = "gas"
        self.haphazard_type = "controlled"
        self.task = "classification"
        self.num_classes = 6

        super().__init__(base_path=base_path, **kwargs)

    def read_data(self, base_path: str = ".") -> tuple[NDArray[np.float64], NDArray[np.int64]]:
        """
        Load and process the Gas dataset.

        Parameters
        ----------
        base_path : str, default="."
            Directory under which the dataset files are searched.

        Returns
        -------
        tuple[NDArray[np.float64], NDArray[np.int64]]
            x : ndarray of shape (n_samples, n_features)
                Feature matrix with dtype float64.
            y : ndarray of shape (n_samples,)
                Integer label vector with dtype int64.

        Raises
        ------
        ValueError
            If any feature vector has length different from 128.
        """
        x_list: list[list[float]] = []
        y_list: list[int] = []

        # Read 10 batch files
        for i in range(1, 11):
            file_path = find_file(base_path, f"gas/batch{i}.dat")
            with open(file_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    label = int(parts[0].split(";")[0]) - 1  # convert to zero-based
                    y_list.append(label)

                    features = [float(p.split(":")[1]) for p in parts[1:]]
                    if len(features) != 128:
                        raise ValueError(
                            f"Feature length mismatch in {file_path}: expected 128, got {len(features)}"
                        )
                    x_list.append(features)

        x = np.array(x_list, dtype=np.float64)
        y = np.array(y_list, dtype=np.int64)
        return x, y


__all__ = ["Gas"]
