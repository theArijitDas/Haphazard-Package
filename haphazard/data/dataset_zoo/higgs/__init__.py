"""
haphazard.data.dataset_zoo.higgs
--------------------------------
higgs dataset implementation for binary classification.
"""

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ...base_dataset import BaseDataset
from ...dataset_zoo import register_dataset
from ....utils.file_utils import find_file


@register_dataset("higgs")
class HIGGS(BaseDataset):
    """
    higgs dataset for binary classification.

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
        Initialize the higgs dataset instance.

        Parameters
        ----------
        base_path : str, default="./"
            Base directory under which the dataset file is located.
        **kwargs
            Additional keyword arguments passed to the parent constructor.
        """
        self.name = "higgs"
        self.haphazard_type = "controlled"
        self.task = "classification"
        self.num_classes = 2

        super().__init__(base_path=base_path, **kwargs)

    def read_data(self, base_path: str = ".") -> tuple[NDArray[np.float64], NDArray[np.int64]]:
        """
        Load and process the higgs dataset.

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
        try:
            data_path = find_file(base_path, "HIGGS_1M.csv.gz")
            df =  pd.read_csv(data_path, compression='gzip')
        except FileNotFoundError:
            data_path = find_file(base_path, "HIGGS_1M.csv")
            df =  pd.read_csv(data_path)

        x = df.iloc[:, 1:].to_numpy(dtype=np.float64)
        y = (df.iloc[:, 0].to_numpy() == 1.0).astype(np.int64)

        return x, y


__all__ = ["HIGGS"]
