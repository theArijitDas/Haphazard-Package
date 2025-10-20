"""
haphazard.models.model_zoo.nb3
------------------------------
Wrapper over NB3 model
"""

import time
from typing import Any

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from .nb3 import NB3
from ...base_model import BaseModel, BaseDataset
from ...model_zoo import register_model
from ....utils.seeding import seed_everything


@register_model("nb3")
class RunNB3(BaseModel):
    """
    Wrapper for running NB3 models within the Haphazard framework.

    Supports binary and multi-class classification tasks.

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
        Initialize the NB3 runner class.

        Parameters
        ----------
        **kwargs
            Optional parameters forwarded to `BaseModel`.
        """
        self.name = "NB3"
        self.tasks = {"classification"}
        self.deterministic = True
        self.hyperparameters = {"fraction_of_features"}

        super().__init__(**kwargs)

    def fit(
        self,
        dataset: BaseDataset,
        mask_params: dict[str, Any] | None = None,
        model_params: dict[str, Any] | None = None,
        seed: int = 42,
    ) -> dict[str, NDArray | float | bool]:
        """
        Run the NB3 model on the given dataset.

        Parameters
        ----------
        dataset : BaseDataset
            Dataset on which the model is trained and evaluated.
        mask_params : dict[str, Any] | None, optional
            Parameters for dataset mask generation.
        model_params : dict[str, Any] | None, optional
            Parameters for NB3 model initialization.
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
        model: NB3 = NB3(num_classes=dataset.num_classes)
        fraction_of_features = model_params.get("fraction_of_features", 1.0)
        num_to_select = max(1, int(fraction_of_features * dataset.n_features))

        if dataset.task == "regression":
            raise NotImplementedError("Regression task not supported for NB3.")

        elif dataset.task == "classification":
            if dataset.num_classes is None:
                raise ValueError(
                    f"For classification task, '{dataset.name}.num_classes' cannot be None."
                )

            pred_list: list[int | float] = []
            logit_list: list[list[float] | float] = []

            start_time = time.perf_counter()

            for i in tqdm(
                range(dataset.n_samples),
                total=dataset.n_samples,
                desc="Running NB3",
            ):
                xi = x[i]
                mi = mask[i]
                yi = y[i]

                document = {k: xi[k] for k in np.where(mi)[0]}
                doc_class = int(yi)

                pred, logit = model.partial_fit(document, doc_class, num_to_select)
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

            # NB3 returns normalized probabilities
            is_logit = False

            return {
                "labels": labels,
                "preds": preds,
                "logits": logits,
                "time_taken": time_taken,
                "is_logit": is_logit,
            }

        # Fallback for unsupported task
        raise ValueError(f"Unknown task type: '{dataset.task}'")


__all__ = ["RunNB3"]
