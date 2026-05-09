"""
haphazard.models.model_zoo.auxnet
----------------------------------
Wrapper over Aux-Net model
"""

import time
from typing import Any

import numpy as np
from numpy.typing import NDArray
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from tqdm import tqdm

from .auxnet import AuxNet, prepare_data_imputation
from ...base_model import BaseModel, BaseDataset
from ....normalization import load_normalizer
from ...model_zoo import register_model
from ....utils.seeding import seed_everything


@register_model("auxnet")
class RunAuxNet(BaseModel):
    """
    Wrapper for running Aux-Net model within the Haphazard framework.

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
        Initialize the Aux-Net runner class.

        Parameters
        ----------
        **kwargs
            Optional parameters forwarded to `BaseModel`.
        """
        self.name = "Aux-Net"
        self.tasks = {"classification"}
        self.deterministic = False
        self.hyperparameters = {
            "lr",
            "no_of_base_layers",
            "no_of_end_layers",
            "nodes_in_each_layer",
            "b",
            "s",
            "n_impute_feat",
        }

        super().__init__(**kwargs)

    def fit(
        self,
        dataset: BaseDataset,
        model_params: dict[str, Any] | None = None,
        seed: int = 42,
    ) -> dict[str, NDArray | float | bool]:
        """
        Run the Aux-Net model on the given dataset.

        Parameters
        ----------
        dataset : BaseDataset
            Dataset on which the model is trained and evaluated.
        model_params : dict[str, Any] | None, optional
            Parameters for Aux-Net model initialization.
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
            raise NotImplementedError("Regression task not supported for Aux-Net.")

        elif dataset.task == "classification":
            if dataset.num_classes is None:
                raise ValueError(
                    f"For classification task, '{dataset.name}.num_classes' cannot be None."
                )

            # Pre-process for Aux-Net
            n_impute_feat = model_params.get("n_impute_feat", 2)
            X_base, X, X_mask = prepare_data_imputation(dataset.x, dataset.mask, 
                                                        imputation_type='forwardfill', 
                                                        n_impute_feat=n_impute_feat)

            base_feat_normalizer = load_normalizer(dataset.norm, num_features=n_impute_feat, replace_with=0.0)

            # Set random seed
            seed_everything(seed)
            model: AuxNet = AuxNet(no_of_aux_layers = X.shape[1], input_shape = X_base.shape[1], 
                                   output_shape = dataset.num_classes, **model_params)
            
            loss_fn = CrossEntropyLoss()
            optimizer = AdamW(model.parameters(), lr=model_params.get("lr", 0.001))

            pred_list: list[int | float] = []
            logit_list: list[list[float] | float] = []

            dataset.x = X
            dataset.mask = X_mask
            
            start_time = time.perf_counter()

            for i, (x, mask, y) in tqdm(
                enumerate(dataset),
                total=dataset.n_samples,
                desc="Running Aux-Net",
            ):
                x_base = base_feat_normalizer(X_base[i], np.ones_like(X_base[i]).astype(bool))

                # Aux-Net cannot handle 'NaN' values inplace of un-observed feature values
                # Replacing un-observed values with 0.0
                x[~mask] = 0.0

                output_logit, partial_logits = model(torch.tensor(x, dtype=torch.float32), 
                                                     torch.tensor(mask, dtype=torch.float32), 
                                                     torch.tensor(x_base, dtype=torch.float32))

                y_one_hot = torch.zeros((dataset.num_classes), dtype=torch.float32)
                y_one_hot[y.item()] = 1.0

                losses_per_layer = []
                for logit in partial_logits:
                    loss = loss_fn(logit, y_one_hot)
                    losses_per_layer.append(loss)
                    loss.backward(retain_graph=True)
                
                optimizer.step()
                model.update_layer_weights(losses_per_layer, mask)
                
                output_logit = output_logit.squeeze().tolist()
                if dataset.num_classes == 2:  # binary classification
                    logit_list.append(output_logit[1])
                else:  # Multi-class classification
                    logit_list.append(output_logit)
                pred_list.append(np.argmax(output_logit).item())

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

            is_logit = True  # Aux-Net outputs raw logits, not normalized probabilities

            return {
                "labels": labels,
                "preds": preds,
                "logits": logits,
                "time_taken": time_taken,
                "is_logit": is_logit,
            }

        # Fallback for unsupported task
        raise ValueError(f"Unknown task type: '{dataset.task}'")


__all__ = ["RunAuxNet"]
