"""
haphazard.normalization.normalizer_zoo.minmax
---------------------------------------------
MinMaxNormalization, online min-max normalization.

This module implements online (incremental) min-max normalization, scaling
each observed feature to a bounded range [0, 1]. It updates feature-wise
minimum and maximum values incrementally and normalizes observed features
accordingly.

The normalized feature is computed as:
    x_norm = (x - min) / (max - min)
"""

from numpy.typing import NDArray
import numpy as np

from ...base_normalizer import BaseNormalizer
from ...normalizer_zoo import register_normalizer


# ----------------------------------------------------------------------
# Min–Max Normalization
# ----------------------------------------------------------------------
@register_normalizer("minmax")
class MinMaxNormalization(BaseNormalizer):
    """
    Online min-max normalization:
        x_norm = (x - min) / (max - min)

    This normalizer tracks running minimum and maximum values for each
    feature and rescales observed features to the range [0, 1]. Designed
    for streaming or incremental data, where feature distributions may
    evolve over time.

    Notes
    -----
    - Handles observed features independently using feature-wise masks.
    - Clips denominator to avoid division by zero.
    - Optionally clips normalized output to [-3, 3] for outlier control.

    Attributes
    ----------
    min_values : NDArray[np.float64]
        Running minimum value per feature.
    max_values : NDArray[np.float64]
        Running maximum value per feature.
    """

    def __init__(self, num_features: int, replace_with: float | str = "nan"):
        super().__init__(num_features, replace_with)
        self.min_values = np.full(num_features, np.inf, dtype=np.float64)
        self.max_values = np.full(num_features, -np.inf, dtype=np.float64)

    def update_params(self, x: NDArray[np.float64], indices: NDArray[np.int64]) -> None:
        """Update feature-wise minimum and maximum values."""
        self.min_values[indices] = np.minimum(self.min_values[indices], x[indices])
        self.max_values[indices] = np.maximum(self.max_values[indices], x[indices])

    def normalize(self, x: NDArray[np.float64], indices: NDArray[np.int64]) -> NDArray[np.float64]:
        """Normalize observed features to [0, 1] using current min and max."""
        denom = self.max_values[indices] - self.min_values[indices]
        denom = np.clip(denom, 1e-6, None)
        x[indices] = (x[indices] - self.min_values[indices]) / denom
        return np.clip(x, -3.0, 3.0)

    def reset(self) -> None:
        """Reset all running min and max values."""
        self.min_values.fill(np.inf)
        self.max_values.fill(-np.inf)
