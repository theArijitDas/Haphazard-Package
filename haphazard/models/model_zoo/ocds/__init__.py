"""
haphazard.models.model_zoo.ocds
-------------------------------
Wrapper over OCDS model
"""

import time
import copy
from typing import Any

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from .ocds import OCDS
from ...base_model import BaseModel, BaseDataset
from ...model_zoo import register_model
from ....utils import seed_everything


class MultiClassWrapper:
    """
    Wraps a binary OCDS model to perform multi-class classification using
    the One-vs-Rest (OvR) strategy.

    Each class is handled by a separate copy of the base binary model.

    Attributes
    ----------
    num_classes : int
        Number of unique classes in the dataset.
    models : list[OCDS]
        Independent OCDS instances for each class.
    """

    def __init__(self, model_instance: OCDS, num_classes: int) -> None:
        """
        Initialize the One-vs-Rest wrapper.

        Parameters
        ----------
        model_instance : OCDS
            A fully initialized binary OLVF model instance.
        num_classes : int
            Total number of output classes.
        """
        self.num_classes: int = num_classes
        self.models: list[OCDS] = [copy.deepcopy(model_instance) for _ in range(num_classes)]

    def partial_fit(
        self,
        X: NDArray[np.float64],
        mask: NDArray[np.bool_],
        Y: NDArray[np.int64],
    ) -> tuple[int, list[float]]:
        """
        Perform a single online update step for one instance.

        Parameters
        ----------
        X : NDArray[np.float64]
            Input feature vector, shape (n_features,).
        mask : NDArray[np.bool_]
            Boolean mask indicationg the available features.
        Y : NDArray[np.int64]
            Ground-truth label in [0, num_classes).

        Returns
        -------
        tuple[int, list[float]]
            - Predicted class index.
            - Logits for each class, length `num_classes`.
        """
        logits = [0.0 for _ in range(self.num_classes)]

        for cls_idx, model in enumerate(self.models):
            binary_label = 1 if Y == cls_idx else 0
            _, logit = model.partial_fit(X, mask, binary_label)
            logits[cls_idx] = float(logit)

        y_pred = int(np.argmax(logits))
        return y_pred, logits


@register_model("ocds")
class RunOCDS(BaseModel):
    """
    Wrapper for running OCDS models within the Haphazard framework.

    Supports binary and multi-class classification tasks. Multi-class
    classification is handled via a One-vs-Rest wrapper.

    Parameters
    ----------
    **kwargs : dict
        Additional arguments to pass to the BaseModel constructor.

    Notes
    -----
    - Supports classification tasks only.
    - For multi-class problems, uses One-vs-Rest strategy internally.
    """

    def __init__(self, **kwargs) -> None:
        """
        Initialize the OCDS runner class.

        Parameters
        ----------
        **kwargs
            Optional parameters forwarded to `BaseModel`.
        """
        self.name = "OCDS"
        self.tasks = {"classification"}
        self.deterministic = True
        self.hyperparameters = {
            "alpha",
            "lamda",
            "beta1",
            "gamma",
            "tau",
            }

        super().__init__(**kwargs)

    def fit(
        self,
        dataset: BaseDataset,
        mask_params: dict[str, Any] | None = None,
        model_params: dict[str, Any] | None = None,
        seed: int = 42,
    ) -> dict[str, NDArray | float | bool]:
        """
        Run the OCDS model on the given dataset.

        Parameters
        ----------
        dataset : BaseDataset
            Dataset on which the model is trained and evaluated.
        mask_params : dict[str, Any] | None, optional
            Parameters for dataset mask generation.
        model_params : dict[str, Any] | None, optional
            Parameters for OCDS model initialization.
        seed : int, default=42
            Random seed for reproducibility.

        Returns
        -------
        dict[str, NDArray | float | bool]
            Dictionary containing:
                - "labels": Ground truth labels.
                - "preds": Predicted labels.
                - "logits": Model output logits or probabilities.
                - "time_taken": Time taken for full dataset pass.
                - "is_logit": Indicates whether scores are logits.
        """
        # --- Validate task type ---
        if dataset.task not in self.tasks:
            raise ValueError(
                f"Model '{self.__class__.__name__}' does not support '{dataset.task}'. "
                f"Supported task(s): {self.tasks}"
            )

        mask_params = mask_params or {}
        model_params = model_params or {}

        # Load data
        x, y = dataset.load_data()
        mask = dataset.load_mask(**mask_params)

        # Set random seed
        seed_everything(seed)
        base_model: OCDS = OCDS(**model_params, 
                                num_features=dataset.n_features)

        if dataset.task == "regression":
            raise NotImplementedError("Regression task not supported for OCDS.")

        elif dataset.task == "classification":
            if dataset.num_classes is None:
                raise ValueError(
                    f"For classification task, '{dataset.name}.num_classes' cannot be None."
                )
            
            if dataset.num_classes == 2:
                model: OCDS | MultiClassWrapper = base_model
            else:
                model = MultiClassWrapper(base_model, num_classes=dataset.num_classes)

            pred_list: list[int | float] = []
            logit_list: list[list[float] | float] = []

            start_time = time.perf_counter()

            for i in tqdm(
                range(dataset.n_samples),
                total=dataset.n_samples,
                desc="Running OCDS",
            ):
                xi = x[i]
                mi = mask[i]
                yi = y[i]

                pred, logit = model.partial_fit(xi, mi, yi)
                pred_list.append(pred)
                logit_list.append(logit)

            end_time = time.perf_counter()
            time_taken = end_time - start_time

            # --- Final formatting ---
            labels = np.asarray(y, dtype=np.int64)
            preds = np.asarray(pred_list, dtype=np.int64)
            logits = np.asarray(logit_list, dtype=np.float64)

            # --- Sanity checks ---
            if dataset.num_classes == 2:
                assert logits.ndim == 1, (
                    f"Expected logits to be 1D for binary classification, got {logits.shape}."
                )
            else:
                assert logits.ndim == 2, (
                    f"Expected logits to be 2D for multi-class classification, got {logits.shape}."
                )

            # OCDS model returns logits
            is_logit = True

            return {
                "labels": labels,
                "preds": preds,
                "logits": logits,
                "time_taken": time_taken,
                "is_logit": is_logit,
            }

        # Fallback for unsupported task
        raise ValueError(f"Unknown task type: '{dataset.task}'")


__all__ = ["RunOCDS"]
