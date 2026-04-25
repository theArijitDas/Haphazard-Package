"""
haphazard.models.model_zoo.v2f
------------------------------
Wrapper over V2F model
"""

import time
from typing import Any

import numpy as np
from numpy.typing import NDArray
from tqdm.auto import tqdm

from .v2f import V2F
from ...base_model import BaseModel, BaseDataset
from ...model_zoo import register_model
from ....utils.seeding import seed_everything


@register_model("v2f")
class RunV2F(BaseModel):
    """
    Wrapper for running V2F model within the Haphazard framework.

    Supports binary and multi-class classification tasks.

    Parameters
    ----------
    **kwargs : dict
        Additional arguments to pass to the BaseModel constructor.

    Notes
    -----
    - Supports classification tasks only.
    """

    def __init__(self, **kwargs) -> None:
        """
        Initialize the V2F runner class.

        Parameters
        ----------
        **kwargs
            Optional parameters forwarded to `BaseModel`.
        """
        self.name = "V2F"
        self.tasks = {"classification"}
        self.deterministic = False
        self.hyperparameters = {
            "embedding_size",
            "aggregation_mode",
            "lr",
            "head",
            "**kwargs"
        }

        super().__init__(**kwargs)

    def fit(
        self,
        dataset: BaseDataset,
        model_params: dict[str, Any] | None = None,
        seed: int = 42,
    ) -> dict[str, NDArray | float | bool]:
        """
        Run the V2F model on the given dataset.

        Parameters
        ----------
        dataset : BaseDataset
            Dataset on which the model is trained and evaluated.
        model_params : dict[str, Any] | None, optional
            Parameters for V2F model initialization.
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

        Note
        ----
        The dataset object now provides normalized, masked samples via __iter__ and __getitem__,
        enabling online (per-instance) learning without full-batch loading.
        Slicing and iteration behavior is controlled by BaseDataset.
        """
        # --- Validate task type ---
        if dataset.task not in self.tasks:
            raise ValueError(
                f"Model '{self.__class__.__name__}' does not support '{dataset.task}'. "
                f"Supported task(s): {self.tasks}"
            )

        model_params = model_params or {}

        if dataset.task == "regression":
            raise NotImplementedError("Regression task not supported for V2F.")

        elif dataset.task == "classification":
            if dataset.num_classes is None:
                raise ValueError(
                    f"For classification task, '{dataset.name}.num_classes' cannot be None."
                )

            # Set random seed
            seed_everything(seed)
            model: V2F = V2F(n_features=dataset.n_features, num_classes=dataset.num_classes, **model_params)

            pred_list: list[int | float] = []
            logit_list: list[list[float] | float] = []

            start_time = time.perf_counter()

            for x, mask, y in tqdm(
                dataset,
                total=dataset.n_samples,
                desc="Running V2F",
            ):

                pred, logit = model(x, mask, y)
                pred_list.append(pred)
                logit_list.append(logit)

            end_time = time.perf_counter()
            time_taken = end_time - start_time

            # --- Final formatting ---
            labels = np.asarray(dataset.y, dtype=np.int64)
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

            is_logit = True  # V2F outputs raw logits, not normalized probabilities

            return {
                "labels": labels,
                "preds": preds,
                "logits": logits,
                "time_taken": time_taken,
                "is_logit": is_logit,
            }

        # Fallback for unsupported task
        raise ValueError(f"Unknown task type: '{dataset.task}'")


__all__ = ["RunV2F"]
