"""
haphazard.models.model_zoo.auxdrop
----------------------------------
Wrapper over Aux-Drop model
"""

import time
from typing import Any

import numpy as np
from numpy.typing import NDArray
from tqdm.auto import tqdm

from .auxdrop import AuxDrop_ODL_AuxLayer1stlayer as AuxDrop
from ...base_model import BaseModel, BaseDataset
from ...model_zoo import register_model
from ....utils.seeding import seed_everything


@register_model("auxdrop")
class RunAuxDrop(BaseModel):
    """
    Wrapper for running Aux-Drop model within the Haphazard framework.

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
        Initialize the Aux-Drop runner class.

        Parameters
        ----------
        **kwargs
            Optional parameters forwarded to `BaseModel`.
        """
        self.name = "Aux-Drop"
        self.tasks = {"classification"}
        self.deterministic = False
        self.hyperparameters = {
            "max_num_hidden_layers",
            "qtd_neuron_per_hidden_layer",
            "n_neuron_aux_layer",
            "b",
            "s",
            "lr",
            "dropout_p",
            "use_cuda",
        }

        super().__init__(**kwargs)

    def fit(
        self,
        dataset: BaseDataset,
        model_params: dict[str, Any] | None = None,
        seed: int = 42,
    ) -> dict[str, NDArray | float | bool]:
        """
        Run the Aux-Drop model on the given dataset.

        Parameters
        ----------
        dataset : BaseDataset
            Dataset on which the model is trained and evaluated.
        model_params : dict[str, Any] | None, optional
            Parameters for Aux-Drop model initialization.
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
            raise NotImplementedError("Regression task not supported for Aux-Drop.")

        elif dataset.task == "classification":
            if dataset.num_classes is None:
                raise ValueError(
                    f"For classification task, '{dataset.name}.num_classes' cannot be None."
                )

            # Set random seed
            seed_everything(seed)
            model: AuxDrop = AuxDrop(n_classes=dataset.num_classes, n_aux_feat=dataset.n_features, **model_params)

            pred_list: list[int | float] = []
            logit_list: list[list[float] | float] = []

            start_time = time.perf_counter()

            for x, mask, y in tqdm(
                dataset,
                total=dataset.n_samples,
                desc="Running Aux-Drop",
            ):
                # Reshaping to appropriate dimensions for AuxDrop
                x = x.reshape(1, -1)
                mask = mask.reshape(1, -1)
                y = y.reshape(-1)

                # Aux-Drop cannot handle 'NaN' values inplace of un-observed feature values
                # Replacing un-observed values with 0.0
                x[~mask] = 0.0

                model.partial_fit(x, mask, y)

            end_time = time.perf_counter()
            time_taken = end_time - start_time

            logits = model.prediction.copy()
            pred_list = [np.argmax(logit).item() for logit in logits]

            if dataset.num_classes == 2:  # binary classification
                logit_list = [logit[1] for logit in logits]
            else:  # Multi-class classification
                logit_list = logits
            
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

            is_logit = True  # Aux-Drop outputs raw logits, not normalized probabilities

            return {
                "labels": labels,
                "preds": preds,
                "logits": logits,
                "time_taken": time_taken,
                "is_logit": is_logit,
            }

        # Fallback for unsupported task
        raise ValueError(f"Unknown task type: '{dataset.task}'")


__all__ = ["RunAuxDrop"]
