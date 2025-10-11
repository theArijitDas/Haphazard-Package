"""
haphazard.utils.metrics
-----------------------
Metrics computation utilities for classification and regression tasks.

This module provides helper functions to evaluate model predictions
for both classification (binary and multi-class) and regression tasks.
It supports standard metrics like accuracy, balanced accuracy, AUROC,
AUPRC, and RMSE, and provides a unified dispatcher for easy evaluation.

Implements
----------
- Classification metrics: accuracy, balanced_accuracy, AUROC, AUPRC, number_of_errors
- Regression metrics: RMSE
- Unified dispatcher: get_all_metrics

Notes
-----
- Classification metrics assume `labels` and `preds` are 1D arrays of shape (n_samples,).
- For multi-class AUROC/AUPRC, logits must be of shape (n_samples, num_classes).
- Regression metrics assume continuous target values.
"""

from typing import Literal, Union

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import (
    roc_auc_score, accuracy_score, average_precision_score,
    balanced_accuracy_score, mean_squared_error
)
from sklearn.preprocessing import label_binarize
from scipy.special import softmax, expit as sigmoid


# -------------------------------------------------------------------------
# Type Definitions
# -------------------------------------------------------------------------
ArrayLike = Union[list, NDArray]
TaskType = Literal["classification", "regression"]

# -------------------------------------------------------------------------
# Classification Metrics
# -------------------------------------------------------------------------
def balanced_accuracy(true: ArrayLike, preds: ArrayLike) -> float:
    """
    Compute the balanced accuracy score.

    Parameters
    ----------
    true : ArrayLike
        Ground truth labels.
    preds : ArrayLike
        Predicted labels.

    Returns
    -------
    float
        Balanced accuracy score.
    """
    return float(balanced_accuracy_score(true, preds))


def accuracy(true: ArrayLike, preds: ArrayLike) -> float:
    """
    Compute standard accuracy score.

    Parameters
    ----------
    true : ArrayLike
        Ground truth labels.
    preds : ArrayLike
        Predicted labels.

    Returns
    -------
    float
        Accuracy score.
    """
    return float(accuracy_score(true, preds))


def auroc(true: ArrayLike, logits: ArrayLike, num_classes: int, is_logit: bool = True) -> float:
    """
    Compute Area Under the Receiver Operating Characteristic (AUROC).

    Parameters
    ----------
    true : ArrayLike
        Ground truth labels.
    logits : ArrayLike
        Model outputs, either raw logits or probabilities.
    num_classes : int
        Number of classes.
    is_logit : bool, default=True
        If True, applies sigmoid/softmax to convert logits to probabilities.

    Returns
    -------
    float
        AUROC score.

    Raises
    ------
    ValueError
        If num_classes < 2.
    AssertionError
        If logits/probs shape does not match expected dimensions.
    """
    true, logits = np.asarray(true), np.asarray(logits)

    if is_logit:
        probs = sigmoid(logits) if num_classes == 2 else softmax(logits, axis=1)
    else:
        probs = logits

    if num_classes == 2:
        assert probs.ndim == 1, f"Expected shape {(true.shape,)}, got {probs.shape}"
        return float(roc_auc_score(true, probs, average="macro"))
    elif num_classes > 2:
        assert probs.ndim == 2, f"Expected shape {(true.shape[0], num_classes)}, got {probs.shape}"
        true_bin = label_binarize(true, classes=np.unique(true))
        return float(roc_auc_score(true_bin, probs, multi_class="ovr"))
    else:
        raise ValueError(f"Invalid num_classes '{num_classes}', expected >= 2.")


def auprc(true: ArrayLike, logits: ArrayLike, num_classes: int, is_logit: bool = True) -> float:
    """
    Compute Area Under the Precision-Recall Curve (AUPRC).

    Parameters
    ----------
    true : ArrayLike
        Ground truth labels.
    logits : ArrayLike
        Model outputs, either raw logits or probabilities.
    num_classes : int
        Number of classes.
    is_logit : bool, default=True
        If True, applies sigmoid/softmax to convert logits to probabilities.

    Returns
    -------
    float
        AUPRC score.

    Raises
    ------
    ValueError
        If num_classes < 2.
    AssertionError
        If logits/probs shape does not match expected dimensions.
    """
    true, logits = np.asarray(true), np.asarray(logits)

    if is_logit:
        probs = sigmoid(logits) if num_classes == 2 else softmax(logits, axis=1)
    else:
        probs = logits

    if num_classes == 2:
        assert probs.ndim == 1, f"Expected shape {(true.shape,)}, got {probs.shape}"
        return float(average_precision_score(true, probs))
    elif num_classes > 2:
        assert probs.ndim == 2, f"Expected shape {(true.shape[0], num_classes)}, got {probs.shape}"
        true_bin = label_binarize(true, classes=np.unique(true))
        return float(average_precision_score(true_bin, probs, average="macro"))
    else:
        raise ValueError(f"Invalid num_classes '{num_classes}', expected >= 2.")


def number_of_errors(true: ArrayLike, preds: ArrayLike) -> int:
    """
    Return the total number of misclassified samples.

    Parameters
    ----------
    true : ArrayLike
        Ground truth labels.
    preds : ArrayLike
        Predicted labels.

    Returns
    -------
    int
        Number of errors.
    """
    true, preds = np.asarray(true), np.asarray(preds)
    return int(np.sum(true.reshape(-1) != preds.reshape(-1)))


# -------------------------------------------------------------------------
# Regression Metrics
# -------------------------------------------------------------------------
def regression_metrics(true: ArrayLike, preds: ArrayLike) -> dict[str, float]:
    """
    Compute regression metrics (currently RMSE).

    Parameters
    ----------
    true : ArrayLike
        Ground truth continuous values.
    preds : ArrayLike
        Predicted continuous values.

    Returns
    -------
    dict[str, float]
        Dictionary with regression metrics:
        {"rmse": float}
    """
    true, preds = np.asarray(true), np.asarray(preds)
    rmse = float(np.sqrt(mean_squared_error(true, preds)))
    return {"rmse": rmse}


# -------------------------------------------------------------------------
# Unified Metrics Dispatcher
# -------------------------------------------------------------------------
def get_all_metrics(
    true: ArrayLike,
    preds: ArrayLike,
    logits: ArrayLike | None = None,
    *,
    task: TaskType = "regression",
    num_classes: int | None = None,
    is_logit: bool = True
) -> dict[str, float]:
    """
    Compute metrics for either regression or classification tasks.

    Parameters
    ----------
    true : ArrayLike
        Ground truth labels or continuous targets.
    preds : ArrayLike
        Predicted labels or continuous values.
    logits : ArrayLike | None, default=None
        Model output logits or probabilities (required for AUROC/AUPRC).
    task : {"classification", "regression"}, default="regression"
        Task type for metrics computation.
    num_classes : int | None, default=None
        Number of classes (used for multi-class metrics if task="classification").
    is_logit : bool, default=True
        Whether `logits` are raw logits (True) or probabilities (False).

    Returns
    -------
    dict[str, float]
        Dictionary of computed metrics.
        - Classification: "acc", "bal_acc", "num_errors", optionally "auroc", "auprc"
        - Regression: "rmse"

    Raises
    ------
    ValueError
        If task type is unsupported.
    """
    true, preds = np.asarray(true), np.asarray(preds)

    if task == "regression":
        return regression_metrics(true, preds)

    elif task == "classification":
        metrics = {
            "bal_acc": balanced_accuracy(true, preds),
            "acc": accuracy(true, preds),
            "num_errors": number_of_errors(true, preds),
        }
        if logits is not None:
            metrics.update({
                "auroc": auroc(true, logits, num_classes or len(np.unique(true)), is_logit=is_logit),
                "auprc": auprc(true, logits, num_classes or len(np.unique(true)), is_logit=is_logit),
            })
        return metrics

    else:
        raise ValueError(f"Unsupported task type '{task}'. Must be 'classification' or 'regression'.")
