"""
haphazard.normalization.base_normalizer
---------------------------------------
Abstract base class for normalizers.

All normalizer implementations must inherit from `BaseNormalizer` and implement both
`update_params()` and `normalize()` methods. '__init__()` method may be overwritten to
initialize required variables.
"""

from abc import ABC, abstractmethod
from numpy.typing import NDArray

import numpy as np


# -------------------------------------------------------------------------
# BaseNormalizer class
# -------------------------------------------------------------------------
class BaseNormalizer(ABC):
    """
    Abstract Base Class for online (incremental) normalization.

    Each subclass must implement:
        - `update_params()`: update internal running statistics
        - `normalize()`: normalize the observed features

    Notes
    -----
    Works with feature-wise masks, updating only the observed indices.
    """

    def __init__(self, num_features: int, replace_with: float | str = "nan"):
        self.num_features = num_features
        self.replacement = np.nan if replace_with == "nan" else float(replace_with)

    # ------------------------------------------------------------------
    @abstractmethod
    def update_params(self, x: NDArray[np.float64], indices: NDArray[np.int64]) -> None:
        """Update internal statistics given the current observed feature indices."""
        raise NotImplementedError

    @abstractmethod
    def normalize(self, x: NDArray[np.float64], indices: NDArray[np.int64]) -> NDArray[np.float64]:
        """Normalize observed features using internal statistics."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    def __call__(self, x: NDArray[np.float64], mask: NDArray[np.bool_]) -> NDArray[np.float64]:
        """
        Apply normalization to the observed features in x.

        Parameters
        ----------
        x : NDArray[np.float64]
            Input feature vector of shape (num_features,).
        mask : NDArray[np.bool_]
            Boolean mask where True indicates observed features.

        Returns
        -------
        NDArray[np.float64]
            Normalized vector with missing features replaced.
        """
        indices = np.where(mask)[0]
        if indices.size == 0:
            return np.full_like(x, self.replacement, dtype=np.float64)

        self.update_params(x, indices)
        x_out = self.normalize(x.copy(), indices)
        x_out[~mask] = self.replacement
        return x_out
    
    # ------------------------------------------------------------------
    def reset(self):
        pass