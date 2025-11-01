"""
haphazard.models.model_zoo.olifl
--------------------------------
Wrapper over OLIFL model for binary and multi-class classification.

Implements `RunOLIFL` runner class and a `MultiClassWrapper` for handling
multi-class tasks using a One-vs-Rest strategy.
"""

import copy
import time
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .olifl import OLIFL
from ...base_model import BaseModel, BaseDataset
from ...model_zoo import register_model


class MultiClassWrapper:
    """
    Wraps a base OLIFL model for multi-class classification using One-vs-Rest (OvR).

    Attributes
    ----------
    num_classes : int
        Number of classes.
    models : list[OLIFL]
        List of independent OLIFL models for each class.
    """

    def __init__(self, base_model: OLIFL, num_classes: int):
        """
        Initialize MultiClassWrapper.

        Parameters
        ----------
        base_model : OLIFL
            Base OLIFL model to be wrapped.
        num_classes : int
            Number of classes.
        """
        self.num_classes: int = num_classes
        self.models: list[OLIFL] = [copy.deepcopy(base_model) for _ in range(num_classes)]

    def fit(
        self,
        x: NDArray[np.float64],
        y: NDArray[np.int64],
        x_hap: NDArray[np.float64],
        mask: NDArray[np.bool_],
    ) -> tuple[list[int], list[list[float]]]:
        """
        Train one OLIFL model per class (OvR) and return predictions and logits.

        Parameters
        ----------
        x : NDArray[np.float64], shape (n_samples, n_features)
            Input features.
        y : NDArray[np.int64], shape (n_samples,)
            Target class labels.
        x_hap : NDArray[np.float64], shape (n_samples, n_features)
            Redundant for signature compatibility.
        mask : NDArray[np.bool_], shape (n_samples, n_features)
            Feature availability mask.

        Returns
        -------
        preds : list[int]
            Predicted class labels.
        logits : list[list[float]]
            Per-class logits for each sample.
        """
        logits: list[list[float]] = []

        for cls_idx in range(self.num_classes):
            model = self.models[cls_idx]
            y_binary = np.where(y == cls_idx, 1, 0)
            _, logits_per_class = model.fit(x, y_binary, x_hap, mask)  # type: ignore
            logits.append(logits_per_class)

        # Stack per-class logits into (n_samples, num_classes)
        logits_array = np.column_stack(logits)
        preds = np.argmax(logits_array, axis=1).tolist()

        return preds, logits_array.tolist()


@register_model("olifl")
class RunOLIFL(BaseModel):
    """
    Runner class for OLIFL model.

    Supports binary and multi-class classification tasks. Multi-class
    classification is handled via One-vs-Rest wrapper.

    Attributes
    ----------
    name : str
        Model name.
    tasks : set[str]
        Supported task types.
    deterministic : bool
        Whether model is deterministic.
    hyperparameters : set[str]
        Supported hyperparameters.
    """

    def __init__(self, **kwargs: Any):
        """
        Initialize the OLIFL runner class.

        Parameters
        ----------
        **kwargs
            Optional parameters forwarded to `BaseModel`.
        """
        self.name = "OLIFL"
        self.tasks = {"classification"}
        self.deterministic = False
        self.hyperparameters = {"C", "option"}

        super().__init__(**kwargs)

    def fit(
        self,
        dataset: BaseDataset,
        model_params: dict[str, Any] | None = None,
        seed: int = 42,
    ) -> dict[str, Any]:
        """
        Run the OLIFL model on a dataset.

        Parameters
        ----------
        dataset : BaseDataset
            Dataset object with `.load_data()` and `.load_mask()`.
        model_params : dict
            Hyperparameters for OLIFL model.
        seed : int, default=42
            Random seed for reproducibility.

        Returns
        -------
        dict[str, Any]
            Dictionary containing:
            - labels : Ground truth labels.
            - preds : Predicted labels.
            - logits : Model output logits or probabilities.
            - time_taken : Time taken for full dataset pass.
            - is_logit : Indicates whether scores are logits (True) or probabilities (False).

        Note
        ----
        The dataset object now provides normalized, masked samples via __iter__ and __getitem__,
        enabling online (per-instance) learning without full-batch loading.
        Slicing and iteration behavior is controlled by BaseDataset.
        """
        # --- Validate task ---
        if dataset.task not in self.tasks:
            raise ValueError(
                f"Model {self.__class__.__name__} does not support {dataset.task}. "
                f"Supported task(s): {self.tasks}"
            )
        
        model_params = model_params or {}

        # Set random seed and initialize base_model
        model_params["seed"] = seed
        base_model: OLIFL = OLIFL(**model_params)

        if dataset.task == "regression":
            raise NotImplementedError("Regression task not supported for OLIFL.")

        elif dataset.task == "classification":
            if dataset.num_classes is None:
                raise ValueError(f"'{dataset.name}.num_classes' cannot be None for classification task.")
            
            if dataset.num_classes == 2:
                model: OLIFL | MultiClassWrapper = base_model
            else:
                print(
                    f"[Running {dataset.num_classes} times "
                    "for multi-class classification (OvR strategy).]"
                )
                model = MultiClassWrapper(base_model, num_classes=dataset.num_classes)
            
            # --- Load normalized data ---
            x_mask_y = [xmy for xmy in dataset]
            x, mask, y = zip(*x_mask_y)
            x, mask, y = map(np.asarray, (x, mask, y))

            # --- Train model ---
            start_time = time.perf_counter()
            preds, logits = model.fit(x, y, x, mask)  # type: ignore
            end_time = time.perf_counter()
            time_taken = end_time - start_time

            # --- Format outputs ---
            labels = np.asarray(y, dtype=np.int64)
            preds = np.asarray(preds, dtype=np.int64)
            logits = np.asarray(logits, dtype=np.float64)

            # --- Sanity checks ---
            if dataset.num_classes == 2:
                assert logits.ndim == 1, (
                    f"Expected logits to be 1D for binary classification, got {logits.shape}."
                )
            else:
                assert logits.ndim == 2, (
                    f"Expected logits to be 2D for multi-class classification, got {logits.shape}."
                )

            is_logit = False  # OLIFL returns probabilities

            return {
                "labels": labels,
                "preds": preds,
                "logits": logits,
                "time_taken": time_taken,
                "is_logit": is_logit,
            }
        
        # Fallback for unsupported task
        raise ValueError(f"Unknown task type: '{dataset.task}'")


__all__ = ["RunOLIFL"]
