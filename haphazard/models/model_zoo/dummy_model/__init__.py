"""
haphazard.models.model_zoo.dummy_model
--------------------------------------
Dummy model for testing.
"""

from typing import Any
import time

import numpy as np
from tqdm import tqdm

from ...base_model import BaseModel, BaseDataset
from ...model_zoo import register_model


@register_model("dummy")
class DummyModel(BaseModel):
    """
    Dummy model for testing and prototyping.

    Supports both classification and regression tasks.
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the DummyModel.

        Notes
        -----
        - Sets model name, supported tasks, determinism, and hyperparameters.
        - Calls BaseModel.__init__ to initialize state.
        """
        self.name = "Dummy"
        self.tasks = {"classification", "regression"}
        self.deterministic = False
        self.hyperparameters = set()

        super().__init__(**kwargs)

    def fit(
        self,
        dataset: BaseDataset,
        mask_params: dict[str, Any] | None = None,
        model_params: dict[str, Any] | None = None,
        seed: int = 42
    ) -> dict[str, Any]:
        """
        Run dummy model on a given dataset.

        Parameters
        ----------
        dataset : BaseDataset
            Dataset on which the model is evaluated.
        mask_params : dict[str, Any] | None
            Parameters for generating a mask. Default is None.
        model_params : dict[str, Any] | None
            Parameters for model initialization. Default is None.
        seed : int, default=42
            Random seed for reproducibility.

        Returns
        -------
        dict[str, Any]
            If `dataset.task == "classification"`:
                - labels : NDArray, shape (n_samples,)
                    Ground truth labels.
                - preds : NDArray, shape (n_samples,)
                    Predicted labels.
                - logits : NDArray, shape (n_samples,) for binary or 
                  (n_samples, num_classes) for multi-class classification
                    Prediction logits or probabilities.
                - time_taken : float
                    Time taken for one dataset pass.
                - is_logit : bool
                    Indicates whether logits are unnormalized (False means probabilities).

            If `dataset.task == "regression"`:
                - targets : NDArray, shape (n_samples,)
                    Ground truth targets.
                - preds : NDArray, shape (n_samples,)
                    Predicted regression values.
                - time_taken : float
                    Time taken for one dataset pass.

        Raises
        ------
        ValueError
            If dataset task is unsupported.
        """
        # Validate that the task is supported
        if dataset.task not in self.tasks:
            raise ValueError(
                f"Model {self.__class__.__name__} does not support '{dataset.task}'. "
                f"Supported tasks: {self.tasks}"
            )

        rng = np.random.default_rng(seed)
        x, y = dataset.load_data()
        mask_params = mask_params or {}
        mask = dataset.load_mask(**mask_params)

        # Regression task
        if dataset.task == "regression":
            pred_list: list[int | float] = []
            start_time = time.perf_counter()

            for x_i, y_i, m_i in tqdm(zip(x, y, mask), total=len(x), desc="Running Dummy Model"):
                pred_list.append(rng.random())

            time_taken = time.perf_counter() - start_time

            return {
                "targets": np.asarray(y, dtype=np.float64),
                "preds": np.asarray(pred_list, dtype=np.float64),
                "time_taken": time_taken,
            }

        # Classification task
        elif dataset.task == "classification":
            pred_list: list[int | float] = []
            prob_list: list[list[float] | float] = []
            start_time = time.perf_counter()

            for x_i, y_i, m_i in tqdm(zip(x, y, mask), total=len(x), desc="Running Dummy Model"):
                if dataset.num_classes == 2:
                    prob = rng.random()
                    prob_list.append(prob)
                    pred_list.append(int(prob >= 0.5))
                else:
                    prob = np.asarray(rng.random(dataset.num_classes))
                    prob_list.append(prob.tolist())
                    pred_list.append(int(np.argmax(prob)))

            time_taken = time.perf_counter() - start_time

            labels = np.asarray(y, dtype=np.int64)
            preds = np.asarray(pred_list, dtype=np.int64)
            logits = np.asarray(pred_list, dtype=np.float64)
            
            return {
                "labels": labels,
                "preds": preds,
                "logits": logits,
                "time_taken": time_taken,
                "is_logit": False,
            }
        
        return {}
