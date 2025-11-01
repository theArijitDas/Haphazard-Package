"""
haphazard.normalization.normalizer_zoo.no_normalization
-------------------------------------------------------
NoNormalization, a pass-through normalizer.

This module defines a baseline normalizer that performs no transformation
on the input features. It simply returns the input as-is, serving as a
fallback or default option when normalization is not required.

Useful for debugging, ablation studies, or when normalization is handled
externally in the preprocessing pipeline.
"""

from numpy.typing import NDArray
import numpy as np

from ...base_normalizer import BaseNormalizer
from ...normalizer_zoo import register_normalizer


# ----------------------------------------------------------------------
# No Normalization
# ----------------------------------------------------------------------
@register_normalizer("none")
class NoNormalization(BaseNormalizer):
    """
    Pass-through normalization (no modification to observed features).

    This normalizer leaves all observed features unchanged while maintaining
    compatibility with the online normalization interface. It is useful when
    normalization is disabled or externally managed.

    Notes
    -----
    - `update_params()` performs no operation.
    - `normalize()` returns the input `x` unchanged.
    """

    def update_params(self, x: NDArray[np.float64], indices: NDArray[np.int64]) -> None:
        """No parameter update (no-op)."""
        pass

    def normalize(self, x: NDArray[np.float64], indices: NDArray[np.int64]) -> NDArray[np.float64]:
        """Return the input features unchanged."""
        return x
