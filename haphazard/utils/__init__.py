"""
haphazard.utils
---------------
Utility module for the Haphazard package.

This module provides general-purpose utilities that assist with:
- Seeding for reproducibility (`seeding.py`)
- File discovery and path utility (`file_utils.py`)
- Metrics computation (`metrics.py`)

Contributors can import helpers directly from this module for convenience:
>>> from haphazard.utils import seed_everything, find_file, get_all_metrics
"""

from .seeding import seed_everything
from .file_utils import find_file
from .metrics import (
    balanced_accuracy,
    accuracy,
    auroc,
    auprc,
    number_of_errors,
    regression_metrics,
    get_all_metrics,
)


# -------------------------------------------------------------------------
# Public exports
# -------------------------------------------------------------------------
__all__ = [
"seed_everything",
"find_file",
"balanced_accuracy",
"accuracy",
"auroc",
"auprc",
"number_of_errors",
"regression_metrics",
"get_all_metrics",
]
