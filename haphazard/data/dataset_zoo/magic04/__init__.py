"""
haphazard.data.dataset_zoo.magic04
----------------------------------
Magic04 dataset implementation for binary classification.
"""

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ...base_dataset import BaseDataset
from ...dataset_zoo import register_dataset
from ....utils.file_utils import find_file


@register_dataset("magic04")
class Magic04(BaseDataset):
    """
    Magic04 dataset for binary classification.

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
    - The raw dataset file contains 10 feature columns and 1 label column.
    - Labels are strings ('g' or 'h'), converted to integer classes:
        'g' → 1, 'h' → 0
    - Data is shuffled for randomized experiments.
    """

    def __init__(self, base_path: str = "./", **kwargs) -> None:
        """
        Initialize the Magic04 dataset instance.

        Parameters
        ----------
        base_path : str, default="./"
            Base directory under which the dataset file is located.
        **kwargs
            Additional keyword arguments passed to the parent constructor.
        """
        self.name = "magic04"
        self.haphazard_type = "controlled"
        self.task = "classification"
        self.num_classes = 2

        super().__init__(base_path=base_path, **kwargs)

    def read_data(self, base_path: str = ".") -> tuple[NDArray[np.float64], NDArray[np.int64]]:
        """
        Load and process the Magic04 dataset.

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
        data_path: str = find_file(base_path, "magic04.data")
        df: pd.DataFrame = pd.read_csv(data_path, sep=",", header=None, engine="python")

        # Shuffle rows to randomize label order
        df = df.sample(frac=1.0, random_state=42)

        # Extract features and labels
        x = df.iloc[:, :10].to_numpy(dtype=np.float64)
        y = (df.iloc[:, 10] == "g").astype(np.int64).to_numpy()

        return x, y


__all__ = ["Magic04"]
