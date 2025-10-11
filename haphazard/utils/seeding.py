"""
haphazard.utils.seeding
-----------------------
Seed management utilities for reproducibility.

This module provides functions to ensure deterministic behavior across
Python's random module, NumPy, and PyTorch (CPU and GPU). It is essential
for reproducible experiments, especially in ML/DL training pipelines.

Notes
-----
- Enabling deterministic behavior may impact GPU performance due to
  disabling some optimizations in CuDNN.
- Use `seed_everything` at the start of your script or experiment runner.
"""

import os
import random

import numpy as np
import torch


# -------------------------------------------------------------------------
# Seeding function
# -------------------------------------------------------------------------
def seed_everything(seed: int) -> None:
    """
    Set seed for reproducibility across random, numpy, and PyTorch (CPU/GPU).

    Parameters
    ----------
    seed : int
        The integer seed to be used for all random number generators.

    Notes
    -----
    - Sets Python's `random` module seed.
    - Sets NumPy's RNG seed.
    - Sets PyTorch CPU and GPU RNG seeds.
    - Forces deterministic operations in CuDNN to improve reproducibility.
    - May slightly reduce GPU performance due to deterministic settings.

    Example
    -------
    >>> seed_everything(42)
    """
    # Python random module
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # NumPy RNG
    np.random.seed(seed)
    
    # PyTorch RNGs
    torch.manual_seed(seed)             # CPU
    torch.cuda.manual_seed(seed)        # Current GPU
    torch.cuda.manual_seed_all(seed)    # All GPUs

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
