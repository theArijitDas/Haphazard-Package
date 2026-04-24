"""
haphazard.data.dataset_zoo.ipd
------------------------------
IPD dataset implementation for binary classification.
"""

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ...base_dataset import BaseDataset
from ...dataset_zoo import register_dataset
from ....utils import find_file


@register_dataset("ipd")
class IPD(BaseDataset):
    """
    IPD dataset for binary classification.

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
    """

    def __init__(self, base_path: str = "./", **kwargs) -> None:
        """
        Initialize the ipd dataset instance.

        Parameters
        ----------
        base_path : str, default="./"
            Base directory under which the dataset file is located.
        **kwargs
            Additional keyword arguments passed to the parent constructor.
        """
        self.name = "ipd"
        self.haphazard_type = "controlled"
        self.task = "classification"
        self.num_classes = 2

        super().__init__(base_path=base_path, **kwargs)

    def read_data(self, base_path: str = ".") -> tuple[NDArray[np.float64], NDArray[np.int64]]:
        """
        Load and process the ipd dataset.

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
        data_path_train: str = find_file(base_path, "ItalyPowerDemand_TRAIN.txt")
        data_path_test: str = find_file(base_path, "ItalyPowerDemand_TEST.txt")

        df_train = pd.read_csv(data_path_train, sep = "  " , header = None, engine = 'python')
        df_test = pd.read_csv(data_path_test, sep = "  " , header = None, engine = 'python')
        df = pd.concat([df_train, df_test])

        label = np.array(df[0] == 1)*1
        df = df.iloc[:,1:]
        df.insert(0, column="class", value=label)
        data = df.sample(frac = 1, random_state=42)

        x = data.iloc[:,1:].to_numpy(dtype=np.float64)
        y = data.iloc[:,:1].to_numpy(dtype=np.int64).ravel()

        return x, y


__all__ = ["IPD"]
