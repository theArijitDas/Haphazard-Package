"""
haphazard.models
----------------
Models Subpackage

Provides the `BaseModel` abstract class that all registered models
must inherit from. Also contains the `model_zoo` subpackage which
maintains all registered models and their loaders.
"""

from .base_model import BaseModel, TaskType
from .model_zoo import load_model, _MODEL_REGISTRY, register_model


# -------------------------------------------------------------------------
# Public exports
# -------------------------------------------------------------------------

__all__ = [
    "BaseModel",
    "TaskType",
    "load_model",
    "_MODEL_REGISTRY",
    "register_model"
]
