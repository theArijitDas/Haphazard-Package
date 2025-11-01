"""
haphazard.models.model_zoo.ovfm
-------------------------------
Wrapper over OVFM model for binary and multi-class classification.

Implements `RunOVFM` runner class and a `MultiClassWrapper` for handling
multi-class tasks using a One-vs-Rest strategy.
"""

import time
import copy
from typing import Any

import numpy as np
from numpy.typing import NDArray
from tqdm.auto import tqdm

from .ovfm import OVFM
from ...base_model import BaseModel, BaseDataset
from ...model_zoo import register_model
from ....utils.seeding import seed_everything


def get_cont_indices(X: NDArray) -> NDArray[np.bool_]:
    max_ord=14
    indices = np.zeros(X.shape[1]).astype(bool)
    for i, col in enumerate(X.T):
        col_nonan = col[~np.isnan(col)]
        col_unique = np.unique(col_nonan)
        if len(col_unique) > max_ord:
            indices[i] = True
    return indices


class MultiClassWrapper:
    """
    Wraps a binary OVFM model to perform multi-class classification using
    the One-vs-Rest (OvR) strategy.

    Each class is handled by a separate copy of the base binary model.

    Attributes
    ----------
    num_classes : int
        Number of unique classes in the dataset.
    models : list[OVFM]
        Independent OVFM instances for each class.
    """

    def __init__(self, model_instance: OVFM, num_classes: int) -> None:
        """
        Initialize the One-vs-Rest wrapper.

        Parameters
        ----------
        model_instance : OVFM
            A fully initialized binary OLVF model instance.
        num_classes : int
            Total number of output classes.
        """
        self.num_classes: int = num_classes
        self.models: list[OVFM] = [copy.deepcopy(model_instance) for _ in range(num_classes)]

    def partial_fit(
        self,
        X: NDArray[np.float64],
        Y: NDArray[np.int64],
    ) -> tuple[int, list[float]]:
        """
        Perform a single online update step for one instance.

        Parameters
        ----------
        X : NDArray[np.float64]
            Input feature vector, shape (n_features,).
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
            _, logit = model.partial_fit(X, binary_label)
            logits[cls_idx] = float(logit)

        y_pred = int(np.argmax(logits))
        return y_pred, logits


@register_model("ovfm")
class RunOVFM(BaseModel):
    """
    Runner class for OVFM model.

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
        Initialize the OVFM runner class.

        Parameters
        ----------
        **kwargs
            Optional parameters forwarded to `BaseModel`.
        """
        self.name = "OVFM"
        self.tasks = {"classification"}
        self.deterministic = False
        self.hyperparameters = {
            "decay_choice", 
            "contribute_error_rate", 
            "decay_coef_change", 
            "batch_size_denominator", 
            }

        super().__init__(**kwargs)

    def fit(
        self,
        dataset: BaseDataset,
        model_params: dict[str, Any] | None = None,
        seed: int = 42,
    ) -> dict[str, Any]:
        """
        Run the OVFM model on the given dataset.

        Parameters
        ----------
        dataset : BaseDataset
            Dataset object with `.load_data()` and `.load_mask()`.
        model_params : dict[str, Any] | None, optional
            Hyperparameters for OVFM model.
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
        # --- Validate task type ---
        if dataset.task not in self.tasks:
            raise ValueError(
                f"Model '{self.__class__.__name__}' does not support '{dataset.task}'. "
                f"Supported task(s): {self.tasks}"
            )

        model_params = model_params or {}

        # OVFM specific pre-processing
        x, mask, _ = zip(*[xmy for xmy in dataset])
        x_masked = np.where(mask, x, np.nan)
        dataset.normalize.reset()

        all_cont_indices = get_cont_indices(x_masked)
        all_ord_indices = ~all_cont_indices
        n_feat = dataset.n_features

        # WINDOW_SIZE is the buffer size. Therefore we set it as the minimum of 10% of total instances of 500 instances.
        WINDOW_SIZE = min(20, int(dataset.n_samples*.1))

        # Set random seed and initialize base_model
        seed_everything(seed)
        base_model: OVFM = OVFM(**model_params, 
                                all_cont_indices=all_cont_indices,
                                all_ord_indices=all_ord_indices,
                                WINDOW_SIZE=WINDOW_SIZE,
                                n_feat=n_feat)

        if dataset.task == "regression":
            raise NotImplementedError("Regression task not supported for OVFM.")

        elif dataset.task == "classification":
            if dataset.num_classes is None:
                raise ValueError(f"'{dataset.name}.num_classes' cannot be None for classification task.")
            
            if dataset.num_classes == 2:
                model: OVFM | MultiClassWrapper = base_model
            else:
                model = MultiClassWrapper(base_model, num_classes=dataset.num_classes)

            pred_list: list[int | float] = []
            logit_list: list[list[float] | float] = []

            # --- Train model ---
            start_time = time.perf_counter()

            for x, _, y in tqdm(
                dataset,
                total=dataset.n_samples,
                desc="Running OVFM",
            ):
                pred, logit = model.partial_fit(x, y)
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

            is_logit = False  # OVFM model returns probabilities

            return {
                "labels": labels,
                "preds": preds,
                "logits": logits,
                "time_taken": time_taken,
                "is_logit": is_logit,
            }

        # Fallback for unsupported task
        raise ValueError(f"Unknown task type: '{dataset.task}'")


__all__ = ["RunOVFM"]
