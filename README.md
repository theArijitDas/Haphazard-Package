# Haphazard

A Python package for **haphazard dataset and model management**.  
Provides a standardized interface for loading datasets (with online normalization) and models, running experiments, and extending with custom datasets or models.

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Datasets](#datasets)
- [Models](#models)
- [Normalization](#normalization)
- [Versions](#versions)
- [Contributing](#contributing)
- [License](#license)

---

## Installation

Install via pip (after packaging):

```bash
pip install haphazard
````

---

## Project Structure

The Haphazard package follows a modular, extensible design:

```
haphazard/
├── __init__.py
├── data/                          # Dataset management
│   ├── __init__.py
│   ├── base_dataset.py
│   ├── mask.py
│   └── datasets/
│       ├── __init__.py
│       ├── dummy_dataset/
│       ├── magic04/
│       ├── a8a/
│       ├── imdb/
│       ├── susy/
│       ├── higgs/
│       ├── dry_bean/
│       └── gas/
├── models/                        # Model management
│   ├── __init__.py
│   ├── base_model.py
│   └── model_zoo/
│       ├── __init__.py
│       ├── dummy_model/
│       ├── dynfo/
│       ├── fae/
│       ├── nb3/
│       ├── ocds/
│       ├── olifl/
│       ├── olvf/
│       ├── orf3v/
│       └── ovfm/
├── normalization/                 # New in v1.1.0
│   ├── __init__.py
│   ├── base_normalizer.py
│   └── normalizer_zoo/
│       ├── __init__.py
│       ├── decimal_scale.py
│       ├── mean.py
│       ├── minmax.py
│       ├── no_normalization.py
│       ├── unit_vector.py
│       └── zscore.py
└── utils/                         # Utilities
    ├── __init__.py
    ├── file_utils.py
    ├── metrics.py
    └── seeding.py
```

### Notes

* `data/base_dataset.py` defines the `BaseDataset` class and integrates normalization support.
* `normalization/base_normalizer.py` defines a universal `BaseNormalizer` base class.
* `normalizer_zoo/` provides built-in normalizers (e.g., **zscore**, **mean**, **decimal_scale**, **no_normalization**).
* `models/base_model.py` defines `BaseModel`, used by all models in `model_zoo/`.
* Dynamic registration of **datasets**, **models**, and now **normalizers** is handled via decorators.

---

## Quick Start

```python
from haphazard import load_dataset, load_model

# Load dataset
dataset = load_dataset("dummy", n_samples=100, n_features=10, norm="zscore")

# Load model
model = load_model("dummy")

# Run model
model_params = {}  # Dummy dataset has no hyperparameters
outputs = model(dataset, model_parameters)
print(outputs)
```

---

## Datasets

* All datasets inherit from `BaseDataset`.
* Example dataset: `DummyDataset`.
* Main interface:

```python
from haphazard import load_dataset
dataset = load_dataset(
   "magic04", 
   base_path="./data", 
   scheme="probabilistic", 
   availability_prob=0.5,
   norm="none"
   )

x, y = dataset.x, dataset.y
mask = dataset.mask
```

### Dataset Attributes

* `name`: str - dataset name
* `task`: `"classification"` | `"regression"`
* `haphazard_type`: `"controlled"` | `"intrinsic"`
* `n_samples`, `n_features`: int
* `num_classes`: int (for classification)
* `normalizer`: optional (default=`"none"`) defines a normalization scheme, if used

### Available Datasets

* Dummy (`"dummy"`)
* Magic04 (`"magic04"`)
* A8a (`"a8a"`)
* IMDB (`"imdb"`)
* Susy (`"susy"`)
* Higgs (`"higgs"`)
* DryBean (`"dry_bean"`)
* Gas (`"gas"`)

---

## Models

* All models inherit from `BaseModel`.
* Example model: `DummyModel`.

```python
from haphazard import load_model
model = load_model("dummy")
model_params = {}  # Hyperparameters of the model
outputs = model(dataset, model_params)
```

### Output

* **Classification**: `labels`, `preds`, `logits`, `time_taken`, `is_logit`
* **Regression**: `targets`, `preds`, `time_taken`

### Available Models

* Dummy - testing/prototyping.
* NB3, FAE - Naive Bayes based models.
* DynFo, ORF3V - Decision stump based models. 
* OLVF, OLIFL, OVFM, OCDS - Linear classifier based models.

---

## Normalization

### Overview

Introduced in **v1.1.0**, the normalization module provides standardized interfaces for **online feature normalization** across datasets and models.

### Using Built-in Normalizers

```python
from haphazard import load_data, load_normalizer

dataset = laod_data(
   "a8a",
   base_path="./",
   scheme="sudden",
   num_chunks=4,
   norm="none"  # No normalization applied internally
)

# Load z-score normalization
normalizer = load_normalizer("zscore", num_features=dataset.n_features)
for x, mask, y in dataset:
   x_norm = normalizer(x, mask)
   ...  
   # Further processing as required

# Load mean normalization
normalizer = load_normalizer("mean", num_features=dataset.n_features)
X, Mask, Y = dataset.x, dataset.y, dataset.mask
for x, mask, y in zip(X, Mask, Y):
   x_norm = normalizer(x, mask)
   ...  
   # Further processing as required
```

or 

```python
from haphazard import load_data, load_normalizer

dataset = laod_data(
   "a8a",
   base_path="./",
   scheme="sudden",
   num_chunks=4,
   norm="zscore"  # Apply normalization internally
)

# Load z-score normalization
# Iterating through the dataset normalizes the input at every step
for x_norm, mask, y in dataset:
   ...  
   # Further processing as required


# Un-normalized values can still be extracted using the following
X, Mask, Y = dataset.x, dataset.y, dataset.mask

# Load mean normalization
normalizer = load_normalizer("mean", num_features=dataset.n_features)
for x, mask, y in zip(X, Mask, Y):
   x_norm = normalizer(x, mask)
   ...  # Further processing as required
```

### Available Normalizers

| Normalizer Name    | Description |
| ------------------ | ----------- |
| `decimal_scale`    | Scales feature values by powers of 10 |
| `mean`             | Online mean normalization by substracting running mean |
| `minmax`           | Scales features by the range (max-min) of the feature value observed |
| `no_normalization` | Pass-through, no normalization applied |
| `unit_vector`      | Normalizes observed values into a unit-vector (scales by L2 norm) |
| `zscore`           | Online mean normalization using running mean and variance (substract mean, scale by variance) |

### Extending Normalization

Developers can register their own normalization schemes-see [Contributing](#contributing).

---

## Versions

### v1.1.4
* Added model **V2F**

### v1.1.3
* Bug fix: **HI2** is not deterministic.
* Set `self.determinitic` = `False` in **HI2** module.

### v1.1.2
* Add model **HI2**

### v1.1.1

* Added model **HapTransformer**
* Update `__repr__` of `base_dataset` 

### v1.1.0

**Major Features**

* Added **Normalization Framework**

  * Introduced new module `normalization/` with base and zoo submodules.
  * Built-in normalizers: `mean`, `zscale`, `no_normalization`, etc.
  * Unified registration via `@register_normalizer`.
  * Datasets and models now support integrated normalization.

**Modifications**

* Updated:

  * `data/base_dataset.py` - normalization integration.
  * `models/base_model.py` - normalization compatibility.
  * `model_zoo` and `datasets` modules - decorator consistency.


### v1.0.9

- Added model **FAE**

- **Bug Fix**
> Update the X2 calculation in NB3 model


### v1.0.8
- Added model **NB3**.


### v1.0.7

- **Bug Fix**
> - Set RunOCDS.determministic = `False` as it uses random initialization.
> - Not passing 'tau' (or passing None) hyperparameter in OCDS will now result in 
> using tau=np.sqrt(1.0/t) as a varied step size, as mentioned in OCDS paper (but not GLSC paper).


### v1.0.6

- Added datasets **A8a**, **IMDB**, **Susy**, and **Higgs**.


### v1.0.5

- Added model **OCDS**.

- **Bug Fixes and Improvements:**
> - In `haphazard/models/model_zoo/dynfo/dynfo.py`:  
>   Updated the `dropLearner()` method to prevent errors when attempting to remove the last remaining weak learner.
>   ```python
>   def dropLearner(self, i):
>       if len(self.learners) == 1:
>           return
>       self.learners.pop(i)
>       self.weights.pop(i)
>       self.acceptedFeatures.pop(i)
>       assert len(self.weights) == len(self.learners) == len(self.acceptedFeatures)
>   ```
>   This ensures stability in low-learner configurations and prevents `IndexError` during runtime.


### v1.0.4

- Added model **ORF3V**.

> NOTE:
>
> * ORF3V also requires an initial buffer, which works similarly to DynFo.
> * ORF3V depends on the optional package `tdigest`, which requires Microsoft Visual C++ Build Tools.
> * To install with this dependency:
>
>   1. Visit: [https://visualstudio.microsoft.com/visual-cpp-build-tools/](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
>   2. Download and install Build Tools for Visual Studio.
>      During installation:
>
>      * Select “Desktop development with C++” workload.
>      * Ensure **MSVC v143 or later**, **Windows 10/11 SDK**, and **CMake tools** are checked.
> * After installation, restart your terminal and re-run:
>
>   ```
>   pip install haphazard[orf3v]
>   ```
>
>   or
>
>   ```
>   pip install haphazard[all]   # installs all optional dependencies
>   ```
> * The package can still be used without installing `tdigest`; only the `ORF3V` model will be unavailable.

- **Bug Fixes and Improvements:**

> - In `haphazard/models/model_zoo/dynfo/__init__.py`: corrected docstring from
>   `"Initialize the OLVF runner class."` -> `"Initialize the DynFo runner class."`
> - In `haphazard/models/model_zoo/dynfo/dynfo.py`: changed
>
>   ```python
>   return int(np.argmax(wc)), float(max(wc))
>   ```
>
>   to
>
>   ```python
>   return int(np.argmax(wc)), float(wc[1])
>   ```
>
>   for correct AUROC/AUPRC compatibility.


### v1.0.3

- Added model **DynFo**

> NOTE:
> - DynFo requires an initial buffer.
> - If no initial buffer size is provided, it is set to 1.
> - The length of the output labels/preds/logits is reduced by the initial buffer size.


### v1.0.2

- Added model **OVFM**


### v1.0.0

(Considered to be the base version, ignore versions before this)

- Includes models **OLVF** and **OLIFL** natively.
- Includes datasets **Magic04**, **Dry Bean** and **Gas**. (Does not include raw files to read from, please use `base_path` argument to point to relevant  path containing the raw files).

---

## Contributing

Haphazard supports easy extensibility for new **datasets**, **models**, and now **normalizers**.

### Adding a new dataset

1. Create a new folder under `haphazard/data/datasets/`, e.g., `my_dataset/`.
2. Add `__init__.py`:

```python
from ...base_dataset import BaseDataset
from ...datasets import register_dataset
import numpy as np

@register_dataset("my_dataset")
class MyDataset(BaseDataset):
    def __init__(self, base_path="./", **kwargs):
        self.name = "my_dataset"
        self.haphazard_type = "controlled"
        self.task = "classification"
        super().__init__(base_path=base_path, **kwargs)

    def read_data(self, base_path="./"):
        # Load or generate x, y
        x = np.random.random((100, 10))
        y = np.random.randint(0, 2, 100)
        return x, y
```

3. The dataset is automatically registered and can be loaded with `load_dataset("my_dataset")`.

### Adding a new model

1. Create a new folder under `haphazard/models/model_zoo/`, e.g., `my_model/`.
2. Add `__init__.py`:

```python
from ...base_model import BaseModel, BaseDataset
from ...model_zoo import register_model
import numpy as np

@register_model("my_model")
class MyModel(BaseModel):
    def __init__(self, **kwargs):
        self.name = "MyModel"
        self.tasks = {"classification", "regression"}
        self.deterministic = True
        self.hyperparameters = set()
        super().__init__(**kwargs)

    def fit(self, dataset: BaseDataset, model_params=None, seed=42):
        # Dummy implementation
        preds = []
        for x, mask, y in dataset:
            preds.append(int(np.random.randint(0, 2)))
        if dataset.task == "classification":
            return {
                "labels": y,
                "preds": preds,
                "logits": preds,
                "time_taken": 0.0,
                "is_logit": True
            }
        elif dataset.task == "regression":
            return {
                "targets": dataset.y,
                "preds": preds,
                "time_taken": 0.0,
            }
```

3. The model is automatically registered and can be loaded with `load_model("my_model")`.


### Adding a New Normalizer

1. Create a folder under:

   ```
   haphazard/normalization/normalizer_zoo/my_normalizer/
   ```

2. Add `__init__.py`:

   ```python
   from ...base_normalizer import OnlineNormalization
   from ...normalizer_zoo import register_normalizer
   import numpy as np
   from numpy.typing import NDArray

   @register_normalizer("my_normalizer")
   class MyNormalizer(OnlineNormalization):
       def __init__(self, num_features: int, replace_with: float | str = "nan"):
           # initialize required parameters
           super().__init__(num_features, replace_with)

       def update_params(self, x: NDArray[np.float64], indices: NDArray[np.int64]) -> None:
           # Update parameters

       def normalize(self, x: NDArray[np.float64], indices: NDArray[np.int64]) -> NDArray[np.float64]:
           # normalize x
           x_norm = ...
           return x_norm
   ```

3. Load dynamically with:

   ```python
   from haphazard import load_normalizer
   normalizer = load_normalizer("my_normalizer", num_features=10)
   ```

---

## License

MIT License.
