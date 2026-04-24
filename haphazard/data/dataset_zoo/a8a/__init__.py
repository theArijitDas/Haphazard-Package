"""
haphazard.data.dataset_zoo.a8a
------------------------------
A8A dataset implementation for binary classification.
"""

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ...base_dataset import BaseDataset
from ...dataset_zoo import register_dataset
from ....utils import find_file


@register_dataset("a8a")
class A8A(BaseDataset):
    """
    A8A dataset for binary classification.

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
        Initialize the a8a dataset instance.

        Parameters
        ----------
        base_path : str, default="./"
            Base directory under which the dataset file is located.
        **kwargs
            Additional keyword arguments passed to the parent constructor.
        """
        self.name = "a8a"
        self.haphazard_type = "controlled"
        self.task = "classification"
        self.num_classes = 2

        super().__init__(base_path=base_path, **kwargs)

    def read_data(self, base_path: str = ".") -> tuple[NDArray[np.float64], NDArray[np.int64]]:
        """
        Load and process the a8a dataset.

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
        data_path: str = find_file(base_path, "a8a.txt")
        df =  pd.read_csv(data_path, sep = " ", header = None, engine = 'python')
        df.dropna(axis=1, how='all', inplace=True)  # 16th column contains only NaN value

        n_features = 123
        n_instances = len(df)
        data = pd.DataFrame(0, index=range(n_instances), columns = list(range(1, n_features+1)))
        
        # Convert labels from {-1, 1} to {1, 0} (1 -> 0, -1 -> 1)
        for j in range(df.shape[0]):
                l = [int(i.split(":")[0])-1 for i in list(df.iloc[j, 1:]) if not pd.isnull(i)]
                data.iloc[j, l] = 1
        label = (df[0].to_numpy() == -1).astype(np.int64)
        data.insert(0, column='class', value=label)

        x = data.iloc[:,1:].to_numpy(dtype=np.float64)          # shape: (32561, 123)
        y = data.iloc[:,:1].to_numpy(dtype=np.int64).ravel()    # shape: (32561,)

        return x, y


__all__ = ["A8A"]
