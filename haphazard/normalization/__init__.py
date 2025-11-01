"""
haphazard.normalization
-----------------------
Normalization Subpackage

Provides the `BaseNormalizer` abstract class that all registered normalizers 
must inherit from. Also contains the `normalizer_zoo` subpackage which
maintains all registered normalizers and their loaders.
"""

from .base_normalizer import BaseNormalizer
from .normalizer_zoo import load_normalizer, _NORMALIZER_REGISTRY, register_normalizer


# -------------------------------------------------------------------------
# Public exports
# -------------------------------------------------------------------------

__all__ = [
    "BaseNormalizer",
    "load_normalizer",
    "_NORMALIZER_REGISTRY",
    "register_normalizer"
]
