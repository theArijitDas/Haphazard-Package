"""
haphazard.data.masks
--------------------
Mask Generation Utilities

This module provides utilities to generate feature observation masks for
datasets exhibiting intrinsic or controlled haphazard patterns.

Supported masking schemes include:
    - "intrinsic"     : Mask derived directly from NaN entries in data.
    - "probabilistic" : Random feature availability with a fixed probability.
    - "sudden"        : Features progressively appear across instance chunks.
    - "obsolete"      : Features progressively disappear across instance chunks.
    - "reappearing"   : Two disjoint feature subsets alternately appear.
"""

from typing import Literal

import numpy as np
from numpy.typing import NDArray


# -------------------------------------------------------------------------
# Type Definitions
# -------------------------------------------------------------------------
MaskScheme = Literal["intrinsic", "probabilistic", "sudden", "obsolete", "reappearing"]


# -------------------------------------------------------------------------
# Mask Validation Helper
# -------------------------------------------------------------------------
def check_mask_each_instance(mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """
    Ensure that every instance has at least one observed feature in the mask.

    Parameters
    ----------
    mask : NDArray[np.bool]
        Boolean array of shape (n_samples, n_features) where
        True indicates observed and False indicates unobserved features.

    Returns
    -------
    : NDArray[np.bool]
        Updated mask of the same shape as input, ensuring that each instance
        (row) contains at least one True value.

    Notes
    -----
    - If an instance has no observed features (all False), one random feature
      is set to True to prevent full-missing samples.
    - Random selection uses NumPy's default RNG with a fixed internal seed
      to ensure reproducibility.
    """
    idx_zero = np.where(mask.sum(axis=1) == 0)[0]
    if idx_zero.size > 0:
        rng = np.random.default_rng(42)
        rand_idx = rng.integers(mask.shape[1], size=idx_zero.size)
        mask[idx_zero, rand_idx] = True
    return mask


# -------------------------------------------------------------------------
# Mask Creation
# -------------------------------------------------------------------------
def create_mask(
    x: NDArray,
    scheme: MaskScheme,
    *,
    availability_prob: float = 0.5,
    num_chunks: int = 5,
    seed: int = 42,
) -> NDArray[np.bool_]:
    """
    Create a boolean mask for input data `x`, based on a chosen masking scheme.

    Parameters
    ----------
    x : NDArray
        Input data array of shape (n_samples, n_features).
    scheme : MaskScheme
        Mask generation scheme, one of:
            - "intrinsic"
            - "probabilistic"
            - "sudden"
            - "obsolete"
            - "reappearing"
    availability_prob : float, default=0.5
        Probability that a feature is observed. Used only for "probabilistic".
    num_chunks : int, default=5
        Number of contiguous instance chunks for chunk-based masking schemes.
    seed : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    : NDArray[np.bool]
        Boolean mask of shape (n_samples, n_features), where True indicates
        a feature is observed.

    Raises
    ------
    ValueError
        If `availability_prob` is not between 0.0 and 1.0.
    ValueError
        If `scheme` is not one of the supported masking types.

    Notes
    -----
    - For "intrinsic", missing values in `x` (NaNs) are directly used to
      infer the mask.
    - All random operations are seeded for deterministic reproducibility.
    """
    n_samples, n_features = x.shape
    mask = np.zeros_like(x, dtype=bool)
    rng = np.random.default_rng(seed)

    # --- Intrinsic Missingness ---
    if scheme == "intrinsic":
        return ~np.isnan(x)

    # --- Probabilistic Missingness ---
    elif scheme == "probabilistic":
        if not (0.0 <= availability_prob <= 1.0):
            raise ValueError(
                f"`availability_prob` must be between 0.0 and 1.0, got {availability_prob!r}"
            )
        mask = rng.random((n_samples, n_features)) < availability_prob

    # --- Sudden Appearance of Features ---
    elif scheme == "sudden":
        samples_per_chunk = max(1, n_samples // num_chunks)
        for i in range(num_chunks):
            end_feat = int(n_features * (i + 1) / num_chunks)
            start_idx = i * samples_per_chunk
            end_idx = min(n_samples, (i + 1) * samples_per_chunk)
            mask[start_idx:end_idx, :end_feat] = True

    # --- Gradual Feature Obsolescence ---
    elif scheme == "obsolete":
        samples_per_chunk = max(1, n_samples // num_chunks)
        for i in range(num_chunks):
            end_feat = int(n_features * (num_chunks - i) / num_chunks)
            start_idx = i * samples_per_chunk
            end_idx = min(n_samples, (i + 1) * samples_per_chunk)
            mask[start_idx:end_idx, :end_feat] = True

    # --- Reappearing Feature Subsets ---
    elif scheme == "reappearing":
        samples_per_chunk = max(1, n_samples // num_chunks)
        mid = n_features // 2
        for i in range(num_chunks):
            start_idx = i * samples_per_chunk
            end_idx = min(n_samples, (i + 1) * samples_per_chunk)
            if i % 2 == 0:
                mask[start_idx:end_idx, :mid] = True
            else:
                mask[start_idx:end_idx, mid:] = True

    else:
        raise ValueError(f"Unknown mask scheme: {scheme!r}")
    
    # Some datasets, not intrinsically haphazard, (like wpbc, wbc) may have some NaN values in them.
    # These feature values are treated as missing even in controlled haphazadness setting.
    mask = mask & ~np.isnan(x)

    # Ensure no instance is fully missing
    return check_mask_each_instance(mask)
