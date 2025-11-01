"""
haphazard.normalization.normalizer_zoo.decimal_scale
----------------------------------------------------
DecimalScaleNormalization — normalization by decimal scaling.

This module implements a simple scaling-based normalization approach that
divides feature values by a power of 10. It is commonly used when feature
ranges are known in advance or require controlled scaling.

The normalized feature is computed as:
    x_norm = x / 10^k
"""

from numpy.typing import NDArray
import numpy as np

from ...base_normalizer import BaseNormalizer
from ...normalizer_zoo import register_normalizer


# ----------------------------------------------------------------------
# Decimal Scale Normalization
# ----------------------------------------------------------------------
@register_normalizer("decimal_scale")
class DecimalScaleNormalization(BaseNormalizer):
    """
    Normalization by decimal scaling:
        x_norm = x / 10^k

    This normalizer divides each observed feature value by a fixed power of
    10 (`scale_power`), effectively shifting the decimal point. It does not
    maintain running statistics and is useful for deterministic scaling or
    bounded numeric inputs.

    Notes
    -----
    - No parameter updates are required.

    Attributes
    ----------
    scale_power : int
        Power of 10 used for scaling.
    """

    def __init__(self, num_features: int, scale_power: int = 3, replace_with: float | str = "nan"):
        super().__init__(num_features, replace_with)
        self.scale_power = scale_power

    def update_params(self, x: NDArray[np.float64], indices: NDArray[np.int64]) -> None:
        """No parameter updates required for decimal scaling."""
        pass

    def normalize(self, x: NDArray[np.float64], indices: NDArray[np.int64]) -> NDArray[np.float64]:
        """Normalize observed features by dividing by 10^scale_power."""
        x[indices] = x[indices] / (10 ** self.scale_power)
        return x
