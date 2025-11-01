"""
haphazard.normalization.normalizer_zoo.unit_vector
--------------------------------------------------
UnitVectorNormalization — L2 normalization to unit length.

This module implements normalization that rescales observed features to
form a unit vector, ensuring the L2 norm equals 1. It is a stateless
normalization method, applied directly on the input without tracking
running statistics.

Useful for models sensitive to feature magnitude (e.g., cosine similarity).
"""

from numpy.typing import NDArray
import numpy as np

from ...base_normalizer import BaseNormalizer
from ...normalizer_zoo import register_normalizer


# ----------------------------------------------------------------------
# Unit Vector Normalization
# ----------------------------------------------------------------------
@register_normalizer("unit_vector")
class UnitVectorNormalization(BaseNormalizer):
    """
    Normalizes observed features to form a unit vector (L2 norm = 1).

    This method scales observed feature values so that their Euclidean norm
    equals 1. It performs no parameter updates and is fully deterministic.

    Notes
    -----
    - No running statistics are maintained.
    """

    def update_params(self, x: NDArray[np.float64], indices: NDArray[np.int64]) -> None:
        """No parameter updates required for unit vector normalization."""
        pass

    def normalize(self, x: NDArray[np.float64], indices: NDArray[np.int64]) -> NDArray[np.float64]:
        """Normalize observed features to have unit L2 norm."""
        vec = x[indices]
        norm = np.linalg.norm(vec)
        if norm > 0:
            x[indices] = vec / norm
        return x
