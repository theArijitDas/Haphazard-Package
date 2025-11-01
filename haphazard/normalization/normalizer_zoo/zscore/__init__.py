"""
haphazard.normalization.normalizer_zoo.zscore
---------------------------------------------
ZScoreNormalization, online z-score normalization.

This module implements online (incremental) z-score normalization using
Welford's algorithm for numerically stable updates of mean and variance.

The normalized feature is computed as:
    x_norm = (x - mean) / std

References
----------
- https://math.stackexchange.com/questions/20593/calculate-variance-from-a-stream-of-sample-values/116344#116344
"""

from numpy.typing import NDArray
import numpy as np

from ...base_normalizer import BaseNormalizer
from ...normalizer_zoo import register_normalizer


# ----------------------------------------------------------------------
# Z-Score Normalization
# ----------------------------------------------------------------------
@register_normalizer("zscore")
class ZScoreNormalization(BaseNormalizer):
    """
    Online z-score normalization:
        x_norm = (x - mean) / std

    This normalizer incrementally updates feature-wise mean and variance using
    Welford's algorithm for numerical stability.

    Notes
    -----
    - Uses unbiased variance estimation (divides by N - 1).
    - Clips normalized values to the range [-3, 3] to limit outliers.

    Attributes
    ----------
    count : NDArray[np.int64]
        Running count of observations per feature.
    mean : NDArray[np.float64]
        Running mean of observed feature values.
    var : NDArray[np.float64]
        Running sum of squared deviations (used to compute variance).
    """
    def __init__(self, num_features: int, replace_with: float | str = "nan"):
        super().__init__(num_features, replace_with)
        self.count = np.zeros(num_features, dtype=np.int64)
        self.mean = np.zeros(num_features, dtype=np.float64)
        self.var = np.zeros(num_features, dtype=np.float64)

    def update_params(self, x: NDArray[np.float64], indices: NDArray[np.int64]) -> None:
        """Update running mean and variance using Welford's algorithm."""
        self.count[indices] += 1
        delta = x[indices] - self.mean[indices]
        self.mean[indices] += delta / self.count[indices]
        delta2 = x[indices] - self.mean[indices]
        self.var[indices] += delta * delta2

    def normalize(self, x: NDArray[np.float64], indices: NDArray[np.int64]) -> NDArray[np.float64]:
        """Normalize observed features using current mean and std estimates."""
        valid = indices[self.count[indices] > 1]
        if valid.size == 0:
            return x
        std = np.sqrt(self.var[valid] / (self.count[valid] - 1))
        std = np.clip(std, 1e-6, None)
        x[valid] = (x[valid] - self.mean[valid]) / std
        return np.clip(x, -3.0, 3.0)

    def reset(self) -> None:
        """Reset all running statistics to zero."""
        self.count.fill(0)
        self.mean.fill(0.0)
        self.var.fill(0.0)