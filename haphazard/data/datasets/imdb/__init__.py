"""
haphazard.data.datasets.imdb
----------------------------
imdb dataset implementation for binary classification.
"""

import warnings
import pickle

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ...base_dataset import BaseDataset
from ...datasets import register_dataset
from ....utils import find_file, seed_everything


@register_dataset("imdb")
class IMDB(BaseDataset):
    """
    imdb dataset for binary classification.

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
    
    Notes:
        - Missing features are represented as -1 in the raw data and converted to np.nan.
        - The original dataset has ordered labels (first 12500 '1', rest '0').
        - Data is shuffled to avoid bias during online training.
    """

    def __init__(self, base_path: str = "./", **kwargs) -> None:
        """
        Initialize the imdb dataset instance.

        Parameters
        ----------
        base_path : str, default="./"
            Base directory under which the dataset file is located.
        **kwargs
            Additional keyword arguments passed to the parent constructor.
        """
        self.name = "imdb"
        self.haphazard_type = "intrinsic"
        self.task = "classification"
        self.num_classes = 2

        super().__init__(base_path=base_path, **kwargs)

    def read_data(self, base_path: str = ".") -> tuple[NDArray[np.float64], NDArray[np.int64]]:
        """
        Load and process the imdb dataset.

        Parameters
        ----------
        base_path : str, default="."
            Directory under which the dataset file is searched.

        Returns
        -------
        tuple[NDArray[np.float64], NDArray[np.int64]]
            x : ndarray of shape (n_samples, n_features)
                Feature matrix with dtype float64.
            y : ndarray of shape (n_samples,)
                Binary label vector (0 or 1) with dtype int64.
        """
        data_path = find_file(base_path, "imdb/imdb")

        # File is saved using older version of numpy. May raise DeprecationWarning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            with open(data_path, "rb") as handle:
                data = pickle.load(handle)

        data = data.astype(float)
        data[data == -1] = np.nan  # Substitute -1 with nan value

        # Random shuffling of dataset
        seed_everything(42)
        np.random.shuffle(data)

        # Rating <7 is negative and >=7 is positive
        label = (data[:, 0] >= 7).astype(np.int64)
        x = data[:, 1:]
        y = label

        return x, y


__all__ = ["IMDB"]
