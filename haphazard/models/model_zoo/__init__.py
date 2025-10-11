"""
haphazard.models.model_zoo
--------------------------
Model registry and dynamic loader.

This module automatically imports all model definitions in the current
directory and provides a convenient `load_model` interface.
"""

import importlib
import pkgutil
from typing import Type, Dict, Any, Callable

from ..base_model import BaseModel


# -------------------------------------------------------------------------
# Model registry
# -------------------------------------------------------------------------
_MODEL_REGISTRY: Dict[str, Type[BaseModel]] = {}


def register_model(name: str) -> Callable[[Type[BaseModel]], Type[BaseModel]]:
    """
    Decorator to register a model class in the global registry.

    Parameters
    ----------
    name : str
        Name of the model. Registration is case-insensitive.

    Returns
    -------
    Callable[[Type[BaseModel]], Type[BaseModel]]
        Decorator that registers the class.

    Usage
    -----
    >>> @register_model("my_model")
    >>> class MyModel(BaseModel):
    >>>     ...
    """
    def decorator(cls: Type[BaseModel]) -> Type[BaseModel]:
        if not issubclass(cls, BaseModel):
            raise TypeError(f"Cannot register '{cls.__name__}': not a subclass of BaseModel.")

        key = name.lower()
        if key in _MODEL_REGISTRY:
            raise ValueError(
                f"Duplicate model name detected: '{key}'. "
                "Note: Registration is case-insensitive."
            )

        _MODEL_REGISTRY[key] = cls
        return cls

    return decorator


# -------------------------------------------------------------------------
# Dynamic import of all submodules
# -------------------------------------------------------------------------
# Automatically import all modules in the current package to register models
_package_name = __name__
for module_info in pkgutil.iter_modules(__path__, prefix=f"{_package_name}."):
    importlib.import_module(module_info.name)


# -------------------------------------------------------------------------
# Public loader
# -------------------------------------------------------------------------
def load_model(name: str, **kwargs: Any) -> BaseModel:
    """
    Load a model by name.

    Parameters
    ----------
    name : str
        Name of the model to load.
    **kwargs : Any
        Additional arguments passed to the model constructor.

    Returns
    -------
    BaseModel
        An instance of the requested model.

    Raises
    ------
    ValueError
        If the model name is not found in the registry.
    """
    key = name.lower()
    if key not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Available models: {list(_MODEL_REGISTRY.keys())}"
        )
    model_cls = _MODEL_REGISTRY[key]
    return model_cls(**kwargs)


# -------------------------------------------------------------------------
# Public exports
# -------------------------------------------------------------------------
__all__ = ["load_model", "_MODEL_REGISTRY", "register_model"]
