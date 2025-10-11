"""
haphazard.data.datasets
-----------------------
Dataset registry and dynamic loader.

This module automatically imports all dataset definitions in the current
directory and provides a convenient `load_dataset` interface.
"""

import importlib
import pkgutil
from typing import Type, Dict, Any, Callable

from ..base_dataset import BaseDataset


# -------------------------------------------------------------------------
# Dataset registry
# -------------------------------------------------------------------------
_DATASET_REGISTRY: Dict[str, Type[BaseDataset]] = {}


def register_dataset(name: str) -> Callable[[Type[BaseDataset]], Type[BaseDataset]]:
    """
    Decorator to register a dataset class in the global registry.

    Parameters
    ----------
    name : str
        The name used to register the dataset (case-insensitive).

    Returns
    -------
    Callable[[Type[BaseDataset]], Type[BaseDataset]]
        A decorator that registers the dataset class in the registry.

    Raises
    ------
    TypeError
        If the decorated class is not a subclass of `BaseDataset`.
    ValueError
        If a dataset with the same name is already registered.

    Examples
    --------
    >>> @register_dataset("my_data")
    ... class MyDataset(BaseDataset):
    ...     pass
    """
    def decorator(cls: Type[BaseDataset]) -> Type[BaseDataset]:
        if not issubclass(cls, BaseDataset):
            raise TypeError(f"Cannot register '{cls.__name__}': not a subclass of BaseDataset.")
        
        key = name.lower()
        if key in _DATASET_REGISTRY:
            raise ValueError(
                f"Duplicate dataset name detected: '{key}'. "
                "(Note: The 'name' used to register a dataset is case-insensitive)"
            )
        
        _DATASET_REGISTRY[key] = cls
        return cls

    return decorator


# -------------------------------------------------------------------------
# Dynamic import of all submodules
# -------------------------------------------------------------------------
# Ensures that all decorated datasets are registered in the global registry.
_package_name = __name__

for module_info in pkgutil.iter_modules(__path__, prefix=f"{_package_name}."):
    importlib.import_module(module_info.name)


# -------------------------------------------------------------------------
# Public loader
# -------------------------------------------------------------------------
def load_dataset(name: str, base_path: str = "./", **kwargs: Any) -> BaseDataset:
    """
    Load a dataset by name.

    Parameters
    ----------
    name : str
        Name of the dataset to load.
    base_path : str, default="./"
        Path to the directory containing the raw data.
    **kwargs : Any
        Additional arguments passed to the dataset constructor.

    Returns
    -------
    BaseDataset
        An instance of the requested dataset.

    Raises
    ------
    ValueError
        If the specified dataset name is not found in the registry.
    """
    key = name.lower()
    if key not in _DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{name}'. "
            f"Available datasets: {list(_DATASET_REGISTRY.keys())}"
        )
    dataset_cls = _DATASET_REGISTRY[key]
    return dataset_cls(base_path, **kwargs)


# -------------------------------------------------------------------------
# Public exports
# -------------------------------------------------------------------------
__all__ = ["load_dataset", "_DATASET_REGISTRY", "register_dataset"]
