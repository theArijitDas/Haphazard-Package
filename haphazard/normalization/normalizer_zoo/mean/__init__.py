"""
haphazard.normalization.normalizer_zoo.mean_normalization
---------------------------------------------------------
MeanNormalization — online mean-centering.

This module implements online mean normalization (centering), which subtracts
the running mean from each observed feature value. It maintains feature-wise
mean estimates incrementally for streaming or online scenarios.

The normalized feature is computed as:
    x_norm = x - mean
"""

from numpy.typing import NDArray
import numpy as np

from ...base_normalizer import BaseNormalizer
from ...normalizer_zoo import register_normalizer


# ----------------------------------------------------------------------
# Mean Normalization
# ----------------------------------------------------------------------
@register_normalizer("mean")
class MeanNormalization(BaseNormalizer):
    """
    Online mean-centering:
        x_norm = x - mean

    This normalizer tracks the running mean of each feature using incremental
    updates. During normalization, the current mean is subtracted from each
    observed feature, effectively centering the data around zero.

    Attributes
    ----------
    count : np.ndarray
        Running count of observations per feature.
    mean : np.ndarray
        Running mean value for each feature.
    """

    def __init__(self, num_features: int, replace_with: float | str = "nan"):
        super().__init__(num_features, replace_with)
        self.count = np.zeros(num_features)
        self.mean = np.zeros(num_features, dtype=np.float64)

    def update_params(self, x: NDArray[np.float64], indices: NDArray[np.int64]) -> None:
        """Incrementally update feature-wise running means."""
        self.count[indices] += 1
        self.mean[indices] += (x[indices] - self.mean[indices]) / self.count[indices]

    def normalize(self, x: NDArray[np.float64], indices: NDArray[np.int64]) -> NDArray[np.float64]:
        """Subtract the current mean from observed features."""
        x[indices] = x[indices] - self.mean[indices]
        return x

    def reset(self) -> None:
        """Reset running means and counts."""
        self.count.fill(0)
        self.mean.fill(0.0)
