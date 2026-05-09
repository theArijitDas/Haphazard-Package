"""
haphazard.models.model_zoo.dynfo
--------------------------------
Wrapper over DynFo model for binary and multi-class classification.

Implements `RunDynFo` runner class.
"""

import time
from typing import Any

import numpy as np
from tqdm import tqdm

from .dynfo import DynFo
from ...base_model import BaseModel, BaseDataset
from ...model_zoo import register_model
from ....utils.seeding import seed_everything


@register_model("dynfo")
class RunDynFo(BaseModel):
    """
    Runner class for DynFo model.

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

    def __init__(self, **kwargs) -> None:
        """
        Initialize the DynFo runner class.

        Parameters
        ----------
        **kwargs
            Optional parameters forwarded to `BaseModel`.
        """
        self.name = "DynFo"
        self.tasks = {"classification"}
        self.deterministic = False
        self.hyperparameters = {
            "alpha",
            "beta",
            "delta",
            "epsilon",
            "gamma",
            "M",
            "N",
            "theta1",
            "theta2",
            "initial_buffer"
        }

        super().__init__(**kwargs)

    def fit(
        self,
        dataset: BaseDataset,
        model_params: dict[str, Any] | None = None,
        seed: int = 42,
    ) -> dict[str, Any]:
        """
        Run the DynFo model on a dataset.

        Parameters
        ----------
        dataset : BaseDataset
            Dataset object.
        model_params : dict[str, Any] | None, optional
            Hyperparameters for DynFo model.
        seed : int, default=42
            Random seed for reproducibility.

        Returns
        -------
        dict[str, Any]
            Dictionary containing:
            - labels: Ground truth labels.
            - preds: Predicted labels.
            - logits: Model output logits or probabilities.
            - time_taken: Time taken for full dataset pass.
            - is_logit: Indicates whether scores are logits (True) or probabilities (False).
        
        Note
        ----
        The dataset object now provides normalized, masked samples via __iter__ and __getitem__,
        enabling online (per-instance) learning without full-batch loading.
        Slicing and iteration behavior is controlled by BaseDataset.
        """
        # --- Validate task ---
        if dataset.task not in self.tasks:
            raise ValueError(
                f"Model '{self.__class__.__name__}' does not support '{dataset.task}'. "
                f"Supported task(s): {self.tasks}"
            )

        model_params = model_params or {}

        if dataset.task == "regression":
            raise NotImplementedError("Regression task not supported for DynFo.")

        elif dataset.task == "classification":
            if dataset.num_classes is None:
                raise ValueError(f"'{dataset.name}.num_classes' cannot be None for classification task.")

            # Get data for initial buffer
            initial_buffer = model_params.pop("initial_buffer", 0)
            initial_buffer = 0 if initial_buffer is None else initial_buffer
            start_idx = 1 if not initial_buffer else initial_buffer

            init_data = dataset[:start_idx]
            if isinstance(init_data, list):
                x, mask, y = zip(*init_data)
                x, mask, y = map(np.asarray, (x, mask, y))
            else:
                x, mask, y = init_data

            # Set random seed and initialize base_model
            seed_everything(seed)
            model: DynFo = DynFo(x, mask, y,
                                 **model_params, num_classes=dataset.num_classes)

            pred_list: list[int | float] = []
            logit_list: list[list[float] | float] = []

            # --- Train model ---
            start_time = time.perf_counter()

            for x, mask, y in tqdm(
                dataset[start_idx:],
                total=dataset.n_samples - start_idx,
                desc="Running DynFo",
            ):
                pred, logit = model.partial_fit(x, mask, int(y))
                pred_list.append(pred)
                logit_list.append(logit)

            end_time = time.perf_counter()
            time_taken = end_time - start_time

            # --- Final formatting ---
            labels = np.asarray(dataset.y[start_idx:], dtype=np.int64)
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

            is_logit = False  # DynFo outputs probability scores, not raw logits

            return {
                "labels": labels,
                "preds": preds,
                "logits": logits,
                "time_taken": time_taken,
                "is_logit": is_logit,
            }

        # Fallback for unsupported task
        raise ValueError(f"Unknown task type: '{dataset.task}'")


__all__ = ["RunDynFo"]
