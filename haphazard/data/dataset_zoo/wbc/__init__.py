"""
haphazard.data.dataset_zoo.wbc
------------------------------
WBC dataset implementation for binary classification.
"""

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ...base_dataset import BaseDataset
from ...dataset_zoo import register_dataset
from ....utils import find_file


@register_dataset("wbc")
class WBC(BaseDataset):
    """
    WBC dataset for binary classification.

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
        Initialize the wbc dataset instance.

        Parameters
        ----------
        base_path : str, default="./"
            Base directory under which the dataset file is located.
        **kwargs
            Additional keyword arguments passed to the parent constructor.
        """
        self.name = "wbc"
        self.haphazard_type = "controlled"
        self.task = "classification"
        self.num_classes = 2

        super().__init__(base_path=base_path, **kwargs)

    def read_data(self, base_path: str = ".") -> tuple[NDArray[np.float64], NDArray[np.int64]]:
        """
        Load and process the wbc dataset.

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
        data_path: str = find_file(base_path, "wbc.data")
        df =  pd.read_csv(data_path, sep = "," , header = None, engine = 'python')
        label = np.array(df[10] == 4)*1
        df = df.iloc[:,1:10]
        set_nan_index = np.where(np.array(df[6]) == '?')
        df.loc[set_nan_index[0], 6] = np.nan
        # df[6][set_nan_index[0]] = np.nan
        df[6] = np.array(df[6]).astype(float)
        df.insert(0, column="class", value=label)
        data = df.sample(frac = 1, random_state=42)

        x = data.iloc[:,1:].to_numpy(dtype=np.float64)
        y = data.iloc[:,:1].to_numpy(dtype=np.int64).ravel()

        return x, y


__all__ = ["WBC"]
