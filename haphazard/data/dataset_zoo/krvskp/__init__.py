"""
haphazard.data.dataset_zoo.krvskp
---------------------------------
KRvsKP dataset implementation for binary classification.
"""

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ...base_dataset import BaseDataset
from ...dataset_zoo import register_dataset
from ....utils import find_file


@register_dataset("krvskp")
class KRvsKP(BaseDataset):
    """
    KRvsKP dataset for binary classification.

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
        Initialize the krvskp dataset instance.

        Parameters
        ----------
        base_path : str, default="./"
            Base directory under which the dataset file is located.
        **kwargs
            Additional keyword arguments passed to the parent constructor.
        """
        self.name = "krvskp"
        self.haphazard_type = "controlled"
        self.task = "classification"
        self.num_classes = 2

        super().__init__(base_path=base_path, **kwargs)

    def read_data(self, base_path: str = ".") -> tuple[NDArray[np.float64], NDArray[np.int64]]:
        """
        Load and process the krvskp dataset.

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
        data_path: str = find_file(base_path, "krvskp.data")
        df =  pd.read_csv(data_path, sep = "," , header = None, engine = 'python')
        label = np.array(df[36] == 'won')*1
        df = df.iloc[:,:36]
        # The different values of the 36 features are - 'b', 'f', 'g', 'l', 'n', 't', 'w'. 
        # Accordingly we assign them the value - 1, 2, 3, 4, 5, 6, 7 resepectively.
        # df[df == 'b'] = 1
        # df[df == 'f'] = 2
        # df[df == 'g'] = 3
        # df[df == 'l'] = 4
        # df[df == 'n'] = 5
        # df[df == 't'] = 6
        # df[df == 'w'] = 7
        # for i in range(df.shape[1] - 1):
        #         df[i] = df[i].astype(float)

        mapping = {'b': 1, 'f': 2, 'g': 3, 'l': 4, 'n': 5, 't': 6, 'w': 7}
        df = df.replace(mapping).astype(float)
        
        df.insert(0, column="class", value=label)
        data = df.sample(frac = 1, random_state=42)

        x = data.iloc[:,1:].to_numpy(dtype=np.float64)
        y = data.iloc[:,:1].to_numpy(dtype=np.int64).ravel()

        return x, y


__all__ = ["KRvsKP"]
