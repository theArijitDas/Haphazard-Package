"""
haphazard.models.model_zoo.haptransformer
-----------------------------------------
Wrapper over HapTransformer model for binary and multi-class classification.

Implements `RunHapTransformer` runner class.
"""

import time
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .haptransformer import hapTransformer
from ...base_model import BaseModel, BaseDataset
from ...model_zoo import register_model
from ....utils import seed_everything


@register_model("haptransformer")
class RunHapTransformer(BaseModel):
    """
    Runner class for HapTransformer model.

    Supports binary and multi-class classification tasks.

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
        Initialize the HapTransformer runner class.

        Parameters
        ----------
        **kwargs
            Optional parameters forwarded to `BaseModel`.
        """
        self.name = "HapTransformer"
        self.tasks = {"classification"}
        self.deterministic = False
        self.hyperparameters = {"hidden_size", "n_heads", "lr"}

        super().__init__(**kwargs)

    def fit(
        self,
        dataset: BaseDataset,
        model_params: dict[str, Any] | None = None,
        seed: int = 42,
    ) -> dict[str, Any]:
        """
        Run the HapTransformer model on a dataset.

        Parameters
        ----------
        dataset : BaseDataset
            Dataset object with `.load_data()` and `.load_mask()`.
        model_params : dict
            Hyperparameters for HapTransformer model.
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
        model_params["n_class"] = dataset.num_classes
        model_params["n_features"] = dataset.n_features
        model_params["device"] = 'cpu'
        model_params["batch_size"] = 1


        if dataset.task == "regression":
            raise NotImplementedError("Regression task not supported for HapTransformer.")

        elif dataset.task == "classification":
            if dataset.num_classes is None:
                raise ValueError(f"'{dataset.name}.num_classes' cannot be None for classification task.")
            
            # Set random seed and initialize base_model
            seed_everything(seed)
            model: hapTransformer = hapTransformer(**model_params)

            # --- Load normalized data ---
            x_mask_y = [xmy for xmy in dataset]
            x, mask, y = zip(*x_mask_y)
            x, mask, y = map(np.asarray, (x, mask, y))
            
            # --- Train model ---
            start_time = time.perf_counter()
            preds, logits = model.fit(x, y, mask)  
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

            is_logit = False  # HapTransformer returns probabilities

            return {
                "labels": labels,
                "preds": preds,
                "logits": logits,
                "time_taken": time_taken,
                "is_logit": is_logit,
            }
        
        # Fallback for unsupported task
        raise ValueError(f"Unknown task type: '{dataset.task}'")


__all__ = ["RunHapTransformer"]
