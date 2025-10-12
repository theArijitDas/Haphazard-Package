"""
haphazard.data.datasets.dry_bean
--------------------------------
Dry Bean dataset implementation for multi-class classification.
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
    Dry Bean dataset for multi-class classification.

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
    - Class labels are mapped from strings to integers.
    - Original dataset has ordered labels as follows:
        Index         -> Label
        0 to 2026     -> 0
        2027 to 3348  -> 1
        3349 to 3870  -> 2
        3871 to 5500  -> 3
        5501 to 7428  -> 4
        7429 to 10064 -> 5
        10065 to 13610 -> 6
    - Data is shuffled for randomized experiments.
    """

    def __init__(self, base_path: str = "./", **kwargs) -> None:
        """
        Initialize the Dry Bean dataset instance.

        Parameters
        ----------
        base_path : str, default="./"
            Base directory under which the dataset file is located.
        **kwargs
            Additional keyword arguments passed to the parent constructor.
        """
        self.name = "dry_bean"
        self.haphazard_type = "controlled"
        self.task = "classification"
        self.num_classes = 7

        super().__init__(base_path=base_path, **kwargs)

    def read_data(self, base_path: str = "./") -> tuple[NDArray[np.float64], NDArray[np.int64]]:
        """
        Load and process the Dry Bean dataset.

        Parameters
        ----------
        base_path : str, default="./"
            Directory under which the dataset file is searched.

        Returns
        -------
        tuple[NDArray[np.float64], NDArray[np.int64]]
            x : ndarray of shape (n_samples, n_features)
                Feature matrix with dtype float64.
            y : ndarray of shape (n_samples,)
                Integer label vector with dtype int64.
        """
        data_path = find_file(base_path, "Dry_Bean_Dataset.arff")
        data, _ = arff.loadarff(data_path)
        df = pd.DataFrame(data)

        # Decode string labels and map to integers
        df["Class"] = df["Class"].str.decode("utf-8")
        class_encoding = {
            "SEKER": 0,
            "BARBUNYA": 1,
            "BOMBAY": 2,
            "CALI": 3,
            "HOROZ": 4,
            "SIRA": 5,
            "DERMASON": 6,
        }
        df["Class"] = df["Class"].map(class_encoding)

        # Shuffle rows for randomized experiments
        df = df.sample(frac=1.0, random_state=42)

        x = df.drop(columns=["Class"]).to_numpy(dtype=np.float64)
        y = df["Class"].to_numpy(dtype=np.int64)

        return x, y


__all__ = ["DryBean"]
