"""
haphazard.models.model_zoo.hi2
------------------------------
Wrapper over HI2 model
"""

import time
from typing import Any

import numpy as np
from numpy.typing import NDArray
from tqdm.auto import tqdm

from .hi2 import HI2
from ...base_model import BaseModel, BaseDataset
from ...model_zoo import register_model
from ....utils.seeding import seed_everything


@register_model("hi2")
class RunHI2(BaseModel):
    """
    Wrapper for running HI2 model within the Haphazard framework.

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
        Initialize the HI2 runner class.

        Parameters
        ----------
        **kwargs
            Optional parameters forwarded to `BaseModel`.
        """
        self.name = "HI2"
        self.tasks = {"classification"}
        self.deterministic = False
        self.hyperparameters = {
            "plot_type",
            "model_name",
            "lr",
            "vert",
            "spacing",
            "device",
        }

        super().__init__(**kwargs)

    def fit(
        self,
        dataset: BaseDataset,
        model_params: dict[str, Any] | None = None,
        seed: int = 42,
    ) -> dict[str, NDArray | float | bool]:
        """
        Run the HI2 model on the given dataset.

        Parameters
        ----------
        dataset : BaseDataset
            Dataset on which the model is trained and evaluated.
        model_params : dict[str, Any] | None, optional
            Parameters for HI2 model initialization.
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
            raise NotImplementedError("Regression task not supported for HI2.")

        elif dataset.task == "classification":
            if dataset.num_classes is None:
                raise ValueError(
                    f"For classification task, '{dataset.name}.num_classes' cannot be None."
                )

            # Set random seed
            seed_everything(seed)
            model: HI2 = HI2(n_class=dataset.num_classes, n_features=dataset.n_features, **model_params, norm=dataset.norm)

            pred_list: list[int | float] = []
            logit_list: list[list[float] | float] = []

            start_time = time.perf_counter()

            for x, mask, y in tqdm(
                dataset,
                total=dataset.n_samples,
                desc="Running HI2",
            ):

                pred, logit = model(x, y, mask)
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

            is_logit = True  # HI2 outputs raw logits, not normalized probabilities

            return {
                "labels": labels,
                "preds": preds,
                "logits": logits,
                "time_taken": time_taken,
                "is_logit": is_logit,
            }

        # Fallback for unsupported task
        raise ValueError(f"Unknown task type: '{dataset.task}'")


__all__ = ["RunHI2"]
