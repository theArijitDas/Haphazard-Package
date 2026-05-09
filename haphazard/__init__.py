"""
haphazard
---------
Top-level imports for main API access.
"""

# Package version
__version__ = "1.2.7"

# User-facing APIs
from .data import load_dataset
from .models import load_model

# Optional APIs for contributors
from .data import BaseDataset, register_dataset
from .models import BaseModel, register_model

# -------------------------------------------------------------------------
# Public exports
# -------------------------------------------------------------------------
__all__ = [
    "load_dataset",
    "load_model",
    "BaseDataset",
    "register_dataset",
    "BaseModel",
    "register_model",
]
