"""
haphazard.models.base_model
---------------------------
Abstract base class for models.

All model implementations must inherit from `BaseModel` and implement both
`__init__()` and `fit()` methods.

Notes
-----
- For classification tasks, `fit()` must always return the keys "logits" and "is_logit".
- This ensures metrics computation can safely distinguish between logits and probabilities.
"""

from abc import ABC, abstractmethod
from typing import Any, Literal, Callable
from functools import wraps

import numpy as np

from ..data import BaseDataset


# -------------------------------------------------------------------------
# Type Definitions
# -------------------------------------------------------------------------
TaskType = Literal["classification", "regression"]


# -------------------------------------------------------------------------
# Utility: decorator for runtime validation of fit outputs
# -------------------------------------------------------------------------
def validate_fit_output(func: Callable[..., dict[str, Any]]) -> Callable[..., dict[str, Any]]:
    """
    Decorator to validate that the output of `fit()` matches the expected keys
    and types for the dataset task.
    """
    @wraps(func)
    def wrapper(self, dataset: BaseDataset, *args, **kwargs) -> dict[str, Any]:
        output = func(self, dataset, *args, **kwargs)

        if dataset.task == "classification":
            # Required keys for classification output
            required_keys = {"labels", "preds", "logits", "time_taken", "is_logit"}
            if not all(key in output for key in required_keys):
                missing = required_keys - output.keys()
                raise KeyError(
                    f"{self.__class__.__name__}.fit() missing keys for classification: {missing}"
                )

            # Validate array types
            array_keys = {"labels", "preds", "logits"}
            for key in array_keys:
                if not isinstance(output[key], np.ndarray):
                    raise TypeError(f"'{key}' must be a numpy ndarray, found {type(output[key])!r}")

            # Validate scalar types
            if not isinstance(output["time_taken"], float):
                raise TypeError(f"'time_taken' must be a float, found {type(output["time_taken"])!r}")
            if not isinstance(output["is_logit"], bool):
                raise TypeError(f"'is_logit' must be a boolean, found {type(output["is_logit"])!r}")

        elif dataset.task == "regression":
            # Required keys for regression output
            required_keys = {"targets", "preds", "time_taken"}
            if not all(key in output for key in required_keys):
                missing = required_keys - output.keys()
                raise KeyError(
                    f"{self.__class__.__name__}.fit() missing keys for regression: {missing}"
                )

            # Validate array types
            if not isinstance(output["targets"], np.ndarray):
                raise TypeError(f"'targets' must be a numpy ndarray, found {type(output["targets"])!r}")
            if not isinstance(output["preds"], np.ndarray):
                raise TypeError(f"'preds' must be a numpy ndarray, found {type(output["preds"])!r}")

            # Validate scalar type
            if not isinstance(output["time_taken"], float):
                raise TypeError(f"'time_taken' must be a float, found {type(output["time_taken"])!r}")

        else:
            raise ValueError(f"Unknown dataset.task: {dataset.task}")

        return output

    return wrapper


# -------------------------------------------------------------------------
# BaseModel class
# -------------------------------------------------------------------------
class BaseModel(ABC):
    """
    Abstract base class for all model classes.

    Attributes
    ----------
    name : str
        Name of the model.
    tasks : set[TaskType]
        Supported task types (classification and/or regression).
    deterministic : bool
        Whether the model is deterministic.
    hyperparameters : set[str]
        Set of model hyperparameter names.

    Notes
    -----
    Subclasses must define the following instance attributes **before**
    calling `super().__init__()`:
        - name
        - tasks
        - deterministic
        - hyperparameters
    """

    # Required subclass attributes
    name: str
    tasks: set[TaskType]
    deterministic: bool
    hyperparameters: set[str]

    # ---------------------------------------------------------------------
    @abstractmethod
    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the model runner class.

        Notes
        -----
        - The initializer must accept arbitrary **kwargs.
        - Subclasses must define required attributes before calling this.
        """
        required_attrs = ["name", "tasks", "deterministic", "hyperparameters"]
        for attr in required_attrs:
            if not hasattr(self, attr):
                raise AttributeError(
                    f"{self.__class__.__name__} must define attribute '{attr}' "
                    "before calling super().__init__()."
                )

        # Store the last used hyperparameters for the model
        self.state_dict: dict[str, Any] = {}

    # ---------------------------------------------------------------------
    @abstractmethod
    @validate_fit_output
    def fit(
        self,
        dataset: BaseDataset,
        mask_params: dict[str, Any] | None = None,
        model_params: dict[str, Any] | None = None,
        seed: int = 42
    ) -> dict[str, Any]:
        """
        Run the model on the given dataset and return results.

        Parameters
        ----------
        dataset : BaseDataset
            Dataset on which the model is trained.
        mask_params : dict[str, Any] | None, optional
            Parameters used for masking the data (default None).
        model_params : dict[str, Any] | None, optional
            Hyperparameters used to initialize the model (default None).
        seed : int, optional
            Random seed for reproducibility (default 42).

        Returns
        -------
        dict[str, Any]
            Classification:
                - labels, NDArray, shape (n_samples,)
                    Ground truth labels.
                - preds, NDArray, shape (n_samples,)
                    Predicted labels.
                - logits, NDArray, shape (n_samples,) or (n_samples, num_classes)
                    Model outputs (logits or probabilities).
                - time_taken, float
                    Time taken to run the model.
                - is_logit, bool
                    True if outputs are raw logits, False if probabilities.

            Regression:
                - targets, NDArray, shape (n_samples,)
                    Ground truth targets.
                - preds, NDArray, shape (n_samples,)
                    Predicted values.
                - time_taken, float
                    Time taken to run the model.
        """
        raise NotImplementedError("Subclasses must implement `fit` method.")

    # ---------------------------------------------------------------------
    def __call__(
        self,
        dataset: BaseDataset,
        mask_params: dict[str, Any] | None = None,
        model_params: dict[str, Any] | None = None,
        seed: int = 42
    ) -> dict[str, Any]:
        """
        Run the model on the given dataset (wrapper over `fit()`).

        Parameters
        ----------
        dataset : BaseDataset
            Dataset on which the model is trained.
        mask_params : dict[str, Any] | None, optional
            Parameters used for masking the data (default None).
        model_params : dict[str, Any] | None, optional
            Hyperparameters used to initialize the model (default None).
        seed : int, optional
            Random seed for reproducibility (default 42).

        Returns
        -------
        dict[str, Any]
            Model outputs, same format as `fit()`.

        Notes
        -----
        - Uses the last stored `state_dict` if `model_params` is None.
        """
        # Use stored hyperparameters if none provided
        if model_params is None:
            model_params = self.state_dict
        if mask_params is None:
            mask_params = {}

        # Ensure required hyperparameters are provided
        if self.hyperparameters and not model_params:
            raise ValueError(
                f"Model '{self.name}' requires hyperparameters {self.hyperparameters}, "
                "but none were provided."
            )

        # Store hyperparameters for future calls
        self.state_dict = model_params

        return self.fit(dataset, mask_params, self.state_dict, seed)

    # ---------------------------------------------------------------------
    def __repr__(self) -> str:
        """Return a concise string representation of the model."""
        return (
            f"{self.__class__.__name__}("
            f"name='{self.name}', "
            f"tasks={self.tasks}, "
            f"deterministic={self.deterministic}, "
            f"hyperparameters={self.hyperparameters})"
        )
