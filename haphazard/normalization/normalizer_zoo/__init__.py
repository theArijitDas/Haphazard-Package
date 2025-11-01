"""
haphazard.normalization.normalizer_zoo
--------------------------------------
Normalizer registry and dynamic loader.

Automatically imports all submodules in this package and provides
a decorator-based interface to register new normalizers.
"""

import importlib
import pkgutil
from typing import Type, Any, Callable

from ..base_normalizer import BaseNormalizer


# -------------------------------------------------------------------------
# Normalizer registry
# -------------------------------------------------------------------------
_NORMALIZER_REGISTRY: dict[str, Type[BaseNormalizer]] = {}


def register_normalizer(name: str) -> Callable[[Type[BaseNormalizer]], Type[BaseNormalizer]]:
    """
    Decorator to register a normalizer class in the global registry.

    Parameters
    ----------
    name : str
        Name of the normalizer. Registration is case-insensitive.

    Returns
    -------
    Callable[[Type[BaseNormalizer]], Type[BaseNormalizer]]
        Decorator that registers the class.

    Usage
    -----
    >>> @register_normalizer("my_normalizer")
    >>> class MyNormalizer(BaseNormalizer):
    >>>     ...
    """
    def decorator(cls: Type[BaseNormalizer]) -> Type[BaseNormalizer]:
        if not issubclass(cls, BaseNormalizer):
            raise TypeError(f"Cannot register '{cls.__name__}': not a subclass of BaseNormalizer.")

        key = name.lower()
        if key in _NORMALIZER_REGISTRY:
            raise ValueError(
                f"Duplicate normalizer name detected: '{key}'. "
                "Note: Registration is case-insensitive."
            )

        _NORMALIZER_REGISTRY[key] = cls
        return cls

    return decorator


# -------------------------------------------------------------------------
# Dynamic import of all submodules
# -------------------------------------------------------------------------
# Automatically import all modules in the current package to register normalizers
_package_name = __name__
for module_info in pkgutil.iter_modules(__path__, prefix=f"{_package_name}."):
    importlib.import_module(module_info.name)


# -------------------------------------------------------------------------
# Public loader
# -------------------------------------------------------------------------
def load_normalizer(name: str, **kwargs: Any) -> BaseNormalizer:
    """
    Load a normalizer by name.

    Parameters
    ----------
    name : str
        Name of the normalizer to load.
    **kwargs : Any
        Additional arguments passed to the normalizer constructor.

    Returns
    -------
    BaseNormalizer
        An instance of the requested normalizer.

    Raises
    ------
    ValueError
        If the normalizer name is not found in the registry.
    """
    key = name.lower()
    if key not in _NORMALIZER_REGISTRY:
        raise ValueError(
            f"Unknown normalizer '{name}'. Available normalizers: {list(_NORMALIZER_REGISTRY.keys())}"
        )
    normalizer_cls = _NORMALIZER_REGISTRY[key]
    return normalizer_cls(**kwargs)


# -------------------------------------------------------------------------
# Public exports
# -------------------------------------------------------------------------
__all__ = ["load_normalizer", "_NORMALIZER_REGISTRY", "register_normalizer"]
