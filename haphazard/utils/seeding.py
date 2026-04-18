"""
haphazard.utils.seeding
-----------------------
Seed management utilities for reproducibility.

This module provides functions to ensure deterministic behavior across
Python's random module, NumPy, PyTorch (CPU and GPU), and scikit-learn
compatible estimators. It is essential for reproducible experiments and
benchmarking pipelines.

Notes
-----
- Enabling deterministic behavior may impact GPU performance due to
  disabling certain CuDNN optimizations.
- Use `seed_everything` at the start of your experiment runner.
- Use `get_seed` to retrieve the active seed for logging or passing to
  scikit-learn estimators via their `random_state` parameter.
"""

import os
import random

import numpy as np
import torch


# -------------------------------------------------------------------------
# Module-level seed state
# -------------------------------------------------------------------------
_ACTIVE_SEED: int | None = None


# -------------------------------------------------------------------------
# Seeding function
# -------------------------------------------------------------------------
def seed_everything(seed: int, deterministic_torch: bool = False) -> None:
    """
    Set seed for reproducibility across random, numpy, and PyTorch (CPU/GPU).

    Parameters
    ----------
    seed : int
        The integer seed to be used for all random number generators.
        Must be a non-negative integer in the range [0, 2**32 - 1].
    deterministic_torch : bool, optional
        If True, enables `torch.use_deterministic_algorithms(True)` for
        stricter determinism guarantees. This may raise errors for PyTorch
        operations that do not have a deterministic implementation, and
        may further reduce GPU performance. Default is False.

    Raises
    ------
    TypeError
        If `seed` is not an integer.
    ValueError
        If `seed` is outside the valid range [0, 2**32 - 1].

    Notes
    -----
    - Sets Python's `random` module seed.
    - Sets the `PYTHONHASHSEED` environment variable.
    - Sets NumPy's RNG seed.
    - Sets PyTorch CPU and GPU RNG seeds.
    - Sets `CUBLAS_WORKSPACE_CONFIG` for full CUDA determinism (PyTorch >= 1.8).
    - Forces deterministic operations in CuDNN.
    - scikit-learn estimators do not use a global RNG; use `get_seed()` to
      pass the active seed via their `random_state` parameter.

    Example
    -------
    >>> seed_everything(42)
    >>> seed_everything(42, deterministic_torch=True)
    """

    # Resolve seed from environment if not explicitly provided
    if seed is None:
        env_seed = os.environ.get('PYTHONHASHSEED')
        if env_seed is None:
            raise ValueError(
                "No seed provided and PYTHONHASHSEED environment variable "
                "is not set. Provide a seed explicitly or set the environment "
                "variable."
            )
        seed = int(env_seed)

    global _ACTIVE_SEED

    # --- Input validation ---
    if isinstance(seed, bool):  # bool subclasses int, so must be checked first
      raise TypeError(
          f"seed must be an integer, got {type(seed).__name__}."
        )
    if not isinstance(seed, int):
        raise TypeError(
            f"seed must be an integer, got {type(seed).__name__}."
        )
    if not (0 <= seed <= 2**32 - 1):
        raise ValueError(
            f"seed must be in the range [0, 2**32 - 1], got {seed}."
        )

    # --- Python stdlib ---
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # --- NumPy ---
    np.random.seed(seed)

    # --- PyTorch ---
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)        # Current GPU
    torch.cuda.manual_seed_all(seed)    # All GPUs

    # Required for full CUDA determinism in PyTorch >= 1.8
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    # CuDNN determinism
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Stricter determinism (opt-in, may raise on unsupported ops)
    if deterministic_torch:
        torch.use_deterministic_algorithms(True)

    # --- Persist active seed ---
    _ACTIVE_SEED = seed


# -------------------------------------------------------------------------
# Seed retrieval
# -------------------------------------------------------------------------
def get_seed() -> int:
    """
    Retrieve the currently active seed set by `seed_everything`.

    This is particularly useful for passing a consistent seed to
    scikit-learn estimators via their `random_state` parameter, or for
    logging the seed used in an experiment run.

    Returns
    -------
    int
        The active seed value.

    Raises
    ------
    RuntimeError
        If `seed_everything` has not been called prior to this function.

    Example
    -------
    >>> seed_everything(42)
    >>> from sklearn.linear_model import SGDClassifier
    >>> clf = SGDClassifier(random_state=get_seed())
    """
    if _ACTIVE_SEED is None:
        raise RuntimeError(
            "`get_seed()` called before `seed_everything()`. "
            "Call `seed_everything(seed)` first."
        )
    return _ACTIVE_SEED