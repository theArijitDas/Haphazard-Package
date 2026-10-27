"""
haphazard.data.base_dataset
---------------------------
Abstract base class for datasets.

All dataset implementations must inherit from `BaseDataset`.
Subclasses must implement read_data() and call super().__init__() from their constructor.
"""

from abc import ABC, abstractmethod
from functools import wraps
from typing import Any, Literal, Callable

import numpy as np
from numpy.typing import NDArray

from .mask import create_mask, MaskScheme
from ..normalization import load_normalizer


# -------------------------------------------------------------------------
# Type Definitions
# -------------------------------------------------------------------------
HaphazardType = Literal["controlled", "intrinsic"]
TaskType = Literal["classification", "regression"]


# -------------------------------------------------------------------------
# Utility: decorator for runtime validation of read_data outputs
# -------------------------------------------------------------------------
def validate_read_data_outputs(
    func: Callable[..., tuple[NDArray, NDArray]]
) -> Callable[..., tuple[NDArray, NDArray]]:
    """
    Decorator to validate that the output of `read_data()` matches the expected format.

    Parameters
    ----------
    func : Callable[..., tuple[NDArray, NDArray]]
        The function to be wrapped, expected to return `(x, y)`.

    Returns
    -------
    Callable[..., tuple[NDArray, NDArray]]
        Wrapped function that validates the output before returning.
    """

    @wraps(func)
    def wrapper(self, *args: Any, **kwargs: Any) -> tuple[NDArray, NDArray]:
        output = func(self, *args, **kwargs)

        # Validate that output is a tuple of length 2
        if not isinstance(output, tuple):
            raise TypeError(f"{func.__name__} must return a tuple, got {type(output)}")
        if len(output) != 2:
            raise ValueError(f"{func.__name__} must return a tuple of length 2, got {len(output)}")

        x, y = output

        # Validate types
        if not isinstance(x, np.ndarray):
            raise TypeError(f"Features x must be a numpy ndarray, got {type(x)}")
        if not isinstance(y, np.ndarray):
            raise TypeError(f"Targets y must be a numpy ndarray, got {type(y)}")

        # Validate shapes
        if x.ndim != 2:
            raise ValueError(f"Features x must be 2D, got shape {x.shape}")
        if y.ndim != 1:
            raise ValueError(f"Targets y must be 1D, got shape {y.shape}")
        if x.shape[0] != y.shape[0]:
            raise ValueError(
                f"Number of samples mismatch: X has {x.shape[0]}, y has {y.shape[0]}"
            )

        return x, y

    return wrapper


# -------------------------------------------------------------------------
# Base Dataset Class
# -------------------------------------------------------------------------
class BaseDataset(ABC):
    """
    Abstract base class for all dataset classes.

    Attributes
    ----------
    name : str
        Name of the dataset.
    haphazard_type : HaphazardType
        Indicates whether the dataset's haphazardness is "controlled" or "intrinsic".
    task : TaskType
        Type of learning task ("classification" or "regression").
    n_features : int
        Number of features in the dataset.
    n_samples : int
        Number of samples in the dataset.
    num_classes : int or None
        Number of output classes if `task='classification'`, otherwise None.

    Notes
    -----
    - All dataset implementations must inherit from this class.
    - Derived classes must implement `read_data()` method 
    and call super().__init__() from their constructor.
    """

    # --- Required metadata fields ---
    name: str
    haphazard_type: HaphazardType
    task: TaskType
    n_features: int
    n_samples: int
    num_classes: int | None

    # -------------------------------------------------------------------------
    def __init__(self, base_path: str = "./", 
                 scheme: MaskScheme = "probabilistic",
                 availability_prob: float = 0.5,
                 num_chunks: int = 5,
                 num_feature_sets: int = 2,
                 overlap: bool|float = False,
                 mask_seed: int = 42,
                 norm: str = "none",
                 **kwargs: Any) -> None:
        """
        Initialize the dataset and load data from disk.

        Parameters
        ----------
        base_path : str, default="./"
            Path to the directory containing the raw data.
        **kwargs : Any
            Additional keyword arguments passed to the dataset loader.

        Notes
        -----
        The initializer must accept `base_path` and arbitrary `**kwargs`.
        """

        # Check required attributes
        required_attrs = ["name", "task", "haphazard_type"]
        for attr in required_attrs:
            if not hasattr(self, attr):
                raise AttributeError(
                    f"{self.__class__.__name__} must define attribute '{attr}' "
                    "before calling super().__init__()."
                )
        
        # Load the data, mask and normalizer
        self.x: NDArray[np.float64]
        self.y: NDArray[np.float64|np.int64]
        self.mask: NDArray[np.bool_]

        if self.haphazard_type=="intrinsic":
            scheme = "intrinsic"

        self.x, self.y = self.read_data(base_path=base_path)
        self.mask = self._load_mask(scheme, 
                                    availability_prob = availability_prob, 
                                    num_chunks = num_chunks, 
                                    num_feature_sets = num_feature_sets,
                                    overlap = overlap,
                                    seed = mask_seed)
        
        # Sanity check
        if not (len(self.x) == len(self.y) == len(self.mask)):
            raise ValueError(
                f"Inconsistent lengths: x={len(self.x)}, y={len(self.y)}, mask={len(self.mask)}"
            )
        
        # Dynamically assign attributes
        self.n_samples, self.n_features = self.x.shape
        if self.task == "regression":
            self.num_classes = None
        elif not getattr(self, "num_classes", None):
            self.num_classes = len(np.unique(self.y))

        self.norm = norm
        self.normalize = load_normalizer(norm, num_features=self.n_features, replace_with=kwargs.get('replace_with', 'nan'))

    # -------------------------------------------------------------------------
    @abstractmethod
    @validate_read_data_outputs
    def read_data(self, base_path: str = "./") -> tuple[NDArray, NDArray]:
        """
        Read the raw dataset file into memory.

        Parameters
        ----------
        base_path : str, default="./"
            Path to the directory containing the raw data.

        Returns
        -------
        x : NDArray
            Feature matrix of shape (n_samples, n_features).
        y : NDArray
            Target vector of shape (n_samples,).

        Raises
        ------
        NotImplementedError
            If not implemented in the subclass.
        """
        raise NotImplementedError("Subclasses must implement `read_data` method.")

    # -------------------------------------------------------------------------
    def _load_mask(
        self,
        scheme: MaskScheme,
        *,
        availability_prob: float,
        num_chunks: int,
        num_feature_sets: int,
        overlap: bool|float,
        seed: int,
    ) -> NDArray[np.bool_]:
        """
        Validate, create, and return a mask for the dataset.

        Parameters
        ----------
        scheme : MaskScheme
            Determines how the mask is generated
            - "intrinsic" : Marks NaNs in the data as missing features.
            - "probabilistic" : Each feature independently observed with probability `availability_prob`.
            - "sudden" : Features become available progressively across chunks of instances.
            - "obsolete" : Features become unavailable progressively across chunks of instances.
            - "reappearing" : Two disjoint feature subsets alternately appear across chunks.
        availability_prob : float, default=0.5
            Probability that a feature is observed (used only for "probabilistic" scheme).
        num_chunks : int, default=5
            Number of chunks for chunk-based schemes.
        seed : int, default=42
            Seed for reproducibility. Passing explicitly is discouraged.

        Returns
        -------
        mask : NDArray[np.bool_]
            Boolean mask of shape (n_samples, n_features), where `True` indicates an available feature.
        """
        mask = create_mask(
            self.x,
            scheme,
            availability_prob=availability_prob,
            num_chunks=num_chunks,
            num_feature_sets = num_feature_sets,
            overlap = overlap,
            seed=seed,
        )

        return mask
    
    # -------------------------------------------------------------------------
    def __len__(self) -> int:
        """Return number of samples in the dataset."""
        return self.n_samples
    
    # -------------------------------------------------------------------------
    def __getitem__(self, i):
        """Return normalized (x, y, mask) tuple or list of tuples for given index or slice."""
        if isinstance(i, slice):
            indices = range(*i.indices(len(self.x)))
            result = []
            for idx in indices:
                result.append((self.normalize(self.x[idx], self.mask[idx]), self.mask[idx], self.y[idx]))
            return result
        else:
            print("process")
            return self.normalize(self.x[i], self.mask[i]), self.mask[i], self.y[i]

    # -------------------------------------------------------------------------
    def __iter__(self):
        """Iterate over samples, returning normalized features, target, and mask."""
        for x, y, mask in zip(self.x, self.y, self.mask):
            yield self.normalize(x, mask), mask, y

    # -------------------------------------------------------------------------
    def __repr__(self) -> str:
        """
        Return a string representation of the dataset.

        Returns
        -------
        : str
            A string summarizing the dataset's metadata and load status.
        """
        data_status = "loaded" if hasattr(self, "x") and self.x is not None else "unloaded"
        return (
            f"{self.__class__.__name__}("
            f"\n\tname='{self.name}', "
            f"\n\ttask='{self.task}', "
            f"\n\thaphazard_type='{self.haphazard_type}', "
            f"\n\tsamples={self.n_samples}, "
            f"\n\tfeatures={self.n_features}, "
            f"\n\tnum_classes={self.num_classes}, "
            f"\n\tdata={data_status}, "
            f"\n\tnorm={self.norm}"
            f"\n)"
        )
