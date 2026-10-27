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

def chunk_bounds(n: int, num_chunks: int) -> list[tuple]:
    """
    Return contiguous (start, end) index pairs that partition n items into
    num_chunks chunks whose sizes differ by at most 1.
    """
    if num_chunks <= 0:
        raise ValueError("num_chunks must be positive.")

    q, r = divmod(n, num_chunks)

    start = 0
    bounds = []

    for i in range(num_chunks):
        size = q + (i < r)
        end = start + size
        bounds.append((start, end))
        start = end

    return bounds


# -------------------------------------------------------------------------
# Mask Creation
# -------------------------------------------------------------------------
def create_mask(
    x: NDArray,
    scheme: MaskScheme,
    *,
    availability_prob: float = 0.5,
    num_chunks: int = 5,
    num_feature_sets: int = 2,
    overlap: bool|float = False,
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
    num_feature_sets: int, default=2
        Number of disjoint sets to divide the features in for "reappearing" scheme.
    overlap: bool|float, default=False
        Denote the fraction of features that overlap (i.e. appear in every chunk) for "reappearing" scheme.
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
        sample_chunks = chunk_bounds(n_samples, num_chunks)
        feature_chunks = chunk_bounds(n_features, num_chunks)
        
        for (sample_start, sample_end), (_, feat_end) in zip(sample_chunks, feature_chunks):
            mask[sample_start:sample_end, :feat_end] = True

    # --- Gradual Feature Obsolescence ---
    elif scheme == "obsolete":
        sample_chunks = chunk_bounds(n_samples, num_chunks)
        feature_chunks = chunk_bounds(n_features, num_chunks)
        
        for (sample_start, sample_end), (feat_start, _) in zip(sample_chunks, feature_chunks):
            mask[sample_start:sample_end, feat_start:] = True

    # --- Reappearing Feature Subsets ---
    elif scheme == "reappearing":
        # -------- Validation --------
        
        if num_feature_sets <= 0:
            raise ValueError("num_feature_sets must be positive.")
        
        if num_feature_sets > num_chunks:
            raise ValueError(
                "num_feature_sets cannot exceed num_chunks because every "
                "feature group must appear at least once."
            )
        
        if overlap is False:
            overlap_frac = 0.0
        
        elif isinstance(overlap, (int, float)) and not isinstance(overlap, bool):
            overlap_frac = float(overlap)
        
            if not (0.0 < overlap_frac < 1.0):
                raise ValueError(
                    "overlap must be False or a float strictly between 0 and 1."
                )
        
        else:
            raise TypeError(
                "overlap must be False or a float strictly between 0 and 1."
            )
        
        n_shared = int(round(overlap_frac * n_features))
        n_remaining = n_features - n_shared
        
        if n_remaining < num_feature_sets:
            raise ValueError(
                f"Only {n_remaining} non-overlapping features remain after "
                f"applying overlap={overlap_frac:.3f}, but "
                f"{num_feature_sets} feature chunks were requested."
            )
        
        # -------- Randomly choose shared features --------
        
        if n_shared:
            perm = rng.permutation(n_features)
        
            shared_features = perm[:n_shared]
            remaining_features = perm[n_shared:]
        else:
            remaining_features = np.arange(n_features)
        
        sample_chunks = chunk_bounds(n_samples, num_chunks)
        feature_chunks = [
            remaining_features[s:e]
            for s, e in chunk_bounds(n_remaining, num_feature_sets)
        ]
        
        # -------- Construct mask --------
        
        for i, (sample_start, sample_end) in enumerate(sample_chunks):
        
            # Shared features (present in every sample chunk)
            if n_shared:
                mask[sample_start:sample_end, shared_features] = True
        
            # Cyclic reappearing feature group
            group = feature_chunks[i % num_feature_sets]
            mask[sample_start:sample_end, group] = True

    else:
        raise ValueError(f"Unknown mask scheme: {scheme!r}")
    
    # Some datasets, not intrinsically haphazard, (like wpbc, wbc) may have some NaN values in them.
    # These feature values are treated as missing even in controlled haphazadness setting.
    mask = mask & ~np.isnan(x)

    # Ensure no instance is fully missing
    return check_mask_each_instance(mask)
