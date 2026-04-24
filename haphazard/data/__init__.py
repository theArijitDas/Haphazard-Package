"""
haphazard.data
--------------
Dataset abstraction and registry subpackage.

This subpackage provides:

- The :class:`BaseDataset` abstract base class, which all datasets must inherit from.
- Dataset registration and loading utilities via the `datasets` submodule.
"""

from .base_dataset import BaseDataset, TaskType, HaphazardType
from .dataset_zoo import load_dataset, _DATASET_REGISTRY, register_dataset


# -------------------------------------------------------------------------
# Public exports
# -------------------------------------------------------------------------
__all__ = [
    "BaseDataset",
    "TaskType",
    "HaphazardType",
    "load_dataset",
    "_DATASET_REGISTRY",
    "register_dataset",
]
