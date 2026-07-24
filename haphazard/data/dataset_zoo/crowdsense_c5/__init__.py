"""
haphazard.data.dataset_zoo.crowdsense_c5
----------------------------------------
Crowdsense_c5 dataset implementation for binary classification.
"""

import pickle
import warnings

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ...base_dataset import BaseDataset
from ...dataset_zoo import register_dataset
from ....utils import find_file


@register_dataset("crowdsense_c5")
class Crowdsense_c5(BaseDataset):
    """
    Crowdsense_c5 dataset for binary classification.

    Attributes
    ----------
    name : str
        Name of the dataset.
    haphazard_type : str
        Indicates dataset type, e.g., "intrinsic".
    task : str
        Task type, e.g., "classification".
    num_classes : int
        Number of unique output classes.
    """

    def __init__(self, base_path: str = "./", **kwargs) -> None:
        """
        Initialize the crowdsense_c5 dataset instance.

        Parameters
        ----------
        base_path : str, default="./"
            Base directory under which the dataset file is located.
        **kwargs
            Additional keyword arguments passed to the parent constructor.
        """
        self.name = "crowdsense_c5"
        self.haphazard_type = "intrinsic"
        self.task = "classification"
        self.num_classes = 2

        super().__init__(base_path=base_path, **kwargs)

    def read_data(self, base_path: str = ".") -> tuple[NDArray[np.float64], NDArray[np.int64]]:
        """
        Load and process the crowdsense_c5 dataset.

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
        data_path: str = find_file(base_path, "crowdsense_c5.pickle")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning, message='.*numpy.core.numeric.*')
            with open(data_path, 'rb') as handle:
                data = pickle.load(handle)
        data = data.astype(float)
        # Substitute each position containing -1 with nan value
        data[data == -1] = np.nan

        # Remove rows that are completely NaN
        data = data[~np.isnan(data[:, 1:]).all(axis=1)]

        # x = data.iloc[:,1:].to_numpy(dtype=np.float64)
        # y = data.iloc[:,:1].to_numpy(dtype=np.int64).ravel()
        
        x = data[:, 1:].astype(np.float64)
        y = data[:,0:1].astype(np.int64).ravel()

        return x, y


__all__ = ["Crowdsense_c5"]
