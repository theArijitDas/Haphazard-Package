"""
haphazard.models.model_zoo.orf3v
--------------------------------
Wrapper over ORF3V model for binary and multi-class classification.

Implements `RunORF3V` runner class.
"""

import time
from typing import Any

import numpy as np
from tqdm import tqdm


from .orf3v import ORF3V
from ...base_model import BaseModel, BaseDataset
from ...model_zoo import register_model
from ....utils.seeding import seed_everything


@register_model("orf3v")
class RunORF3V(BaseModel):
    """
    Runner class for ORF3V model.

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
        Initialize the ORF3V runner class.

        Parameters
        ----------
        **kwargs
            Optional parameters forwarded to `BaseModel`.
        """
        self.name = "ORF3V"
        self.tasks = {"classification"}
        self.deterministic = False
        self.hyperparameters = {
            "forestSize",
            "replacementInterval",
            "replacementChance",
            "windowSize",
            "updateStrategy",
            "alpha",
            "delta",
        }

        super().__init__(**kwargs)

    def fit(
        self,
        dataset: BaseDataset,
        mask_params: dict[str, Any] | None = None,
        model_params: dict[str, Any] | None = None,
        seed: int = 42,
    ) -> dict[str, Any]:
        """
        Run the DynDo model on a dataset.

        Parameters
        ----------
        dataset : BaseDataset
            Dataset object with `.load_data()` and `.load_mask()`.
        mask_params : dict[str, Any] | None, optional
            Parameters for dataset mask generation.
        model_params : dict[str, Any] | None, optional
            Hyperparameters for DynDo model.
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
        """
        # --- Validate task ---
        if dataset.task not in self.tasks:
            raise ValueError(
                f"Model '{self.__class__.__name__}' does not support '{dataset.task}'. "
                f"Supported task(s): {self.tasks}"
            )

        mask_params = mask_params or {}
        model_params = model_params or {}

        # --- Load data ---
        x, y = dataset.load_data()
        mask = dataset.load_mask(**mask_params)

        if dataset.task == "regression":
            raise NotImplementedError("Regression task not supported for OLVF.")

        elif dataset.task == "classification":
            if dataset.num_classes is None:
                raise ValueError(f"'{dataset.name}.num_classes' cannot be None for classification task.")

            # Set random seed and initialize base_model
            initial_buffer = model_params.pop("initial_buffer", 0)
            initial_buffer = 0 if initial_buffer is None else initial_buffer
            start_idx = 1 if not initial_buffer else initial_buffer

            seed_everything(seed)
            model: ORF3V = ORF3V(x[:start_idx], mask[:start_idx], y[:start_idx],
                                 **model_params, numClasses=dataset.num_classes)

            pred_list: list[int | float] = []
            logit_list: list[list[float] | float] = []

            # --- Train model ---
            start_time = time.perf_counter()

            for i in tqdm(
                range(start_idx, dataset.n_samples),
                total=dataset.n_samples - initial_buffer,
                desc="Running ORF3V",
            ):
                xi, mi, yi = x[i], mask[i], y[i]
                pred, logit = model.partial_fit(xi, mi, int(yi))
                pred_list.append(pred)
                logit_list.append(logit)

            end_time = time.perf_counter()
            time_taken = end_time - start_time

            # --- Final formatting ---
            labels = np.asarray(y[start_idx:], dtype=np.int64)
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

            is_logit = False  # ORF3V model returns probabilities

            return {
                "labels": labels,
                "preds": preds,
                "logits": logits,
                "time_taken": time_taken,
                "is_logit": is_logit,
            }

        # Fallback for unsupported task
        raise ValueError(f"Unknown task type: '{dataset.task}'")


__all__ = ["RunORF3V"]
