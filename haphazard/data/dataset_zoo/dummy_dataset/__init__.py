"""
haphazard.data.dataset_zoo.dummy_dataset
----------------------------------------
Implements a dummy dataset for testing and prototyping.
"""

import numpy as np
from numpy.typing import NDArray

from ...base_dataset import BaseDataset, TaskType
from ...dataset_zoo import register_dataset


@register_dataset("dummy")
class DummyDataset(BaseDataset):
    """
    Dummy dataset for testing and validation.

    This dataset generates synthetic features and targets for quick
    prototyping, unit testing, or pipeline validation.
    """

    def __init__(
        self,
        base_path: str = "./",
        n_samples: int = 100,
        n_features: int = 10,
        task: TaskType = "classification",
        num_classes: int = 2,
        **kwargs
    ) -> None:
        """
        Initialize the dummy dataset.

        Parameters
        ----------
        base_path : str, default="./"
            Base path to the dataset (unused here).
        n_samples : int, default=100
            Number of samples to generate.
        n_features : int, default=10
            Number of features per sample.
        task : {"classification", "regression"}, default="classification"
            Task type that determines the target generation scheme.
        num_classes : int, default=2
            Number of output classes (used only for classification tasks).
        **kwargs : Any
            Additional arguments passed to :class:`BaseDataset`.
        """
        self.name = "dummy"
        self.haphazard_type = "controlled"
        self.task = task
        self._n_samples = n_samples
        self._n_features = n_features
        self.num_classes = num_classes if task == "classification" else None

        super().__init__(base_path=base_path, **kwargs)

    def read_data(self, base_path: str = "./") -> tuple[NDArray, NDArray]:
        """
        Generate dummy feature and target data.

        Parameters
        ----------
        base_path : str, default="./"
            Base path (unused in this dataset).

        Returns
        -------
        tuple of (NDArray, NDArray)
            Feature matrix `x` of shape (n_samples, n_features)
            and target vector `y` of shape (n_samples,).
        """
        x = np.random.random((self._n_samples, self._n_features)).astype(np.float64)

        if self.task == "classification":
            y = np.random.randint(0, self.num_classes, size=self._n_samples, dtype=np.int64)
        else:
            y = np.random.random((self._n_samples)).astype(np.float64)

        return x, y


__all__ = ["DummyDataset"]
