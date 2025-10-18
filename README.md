# Haphazard

A Python package for **haphazard dataset and model management**.  
Provides a standardized interface for loading datasets and models, running experiments, and extending with custom datasets or models.

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Datasets](#datasets)
- [Models](#models)
- [Versions](#versions)
- [Contributing](#contributing)
- [License](#license)

---

## Installation

Install via pip (after packaging):

```bash
pip install haphazard
````

Or for local development:

```bash
git clone <repo_url>
cd haphazard
pip install -e .
```

---

## Project Structure

The Haphazard package has a modular layout:

```
haphazard/
├── __init__.py                    # Top-level package
├── data/                          # Dataset related modules
│   ├── __init__.py
│   ├── base_dataset.py            # Abstract BaseDataset class
│   ├── mask.py
│   └── datasets/   
│       ├── __init__.py            # All dataset implementations
│       ├── dummu_dataset/
│       ├── magic04/
│       ├── dry_bean/
│       └── gas/
├── models/                        # Model related modules
│   ├── __init__.py
│   ├── base_model.py              # Abstract BaseModel class
│   └── model_zoo/                 # All model implementations
│       ├── __init__.py
│       ├── dummu_model/
│       ├── olvf/
│       ├── olifl/
│       └── ovfm/
└── utils/                         # Optional helper functions
    └── ...
```

**Notes:**

* `data/base_dataset.py` defines `BaseDataset`.
* `data/datasets/` contains registered datasets; each dataset is a submodule with `__init__.py`.
* `models/base_model.py` defines `BaseModel`.
* `models/model_zoo/` contains registered models; each model is a submodule with `__init__.py`.
* `utils/` is optional, for shared helpers.

This layout allows **dynamic registration** of datasets and models via decorators.

---


## Quick Start

```python
from haphazard import load_dataset, load_model

# Load dataset
dataset = load_dataset("dummy", n_samples=100, n_features=10)

# Load model
model = load_model("dummy")

# Run model
outputs = model(dataset)
print(outputs)
```

---

## Datasets

* All datasets must inherit from `BaseDataset`.
* Example dataset: `DummyDataset`.
* Main interface:

```python
from haphazard import load_dataset

dataset = load_dataset("dummy", base_path="./data")
x, y = dataset.load_data()
mask = dataset.load_mask(scheme="probabilistic", availability_prob=0.5)
```

### Dataset Attributes

* `name` : str — Dataset name.
* `task` : `"classification"` | `"regression"`.
* `haphazard_type` : `"controlled"` | `"intrinsic"`.
* `n_samples`, `n_features` : int.
* `num_classes` : int (for classification).

### Available Datasets (does not include raw files)

* Dummy ("dummy"): For testing and prototyping.
* Magic04 ("magic04"): Binary classification
* Dry Bean ("dry_bean"): Multi-class classification
* GAS ("gas"): Multi-class classification

---

## Models

* All models must inherit from `BaseModel`.
* Example model: `DummyModel`.
* Main interface:

```python
from haphazard import load_model

model = load_model("dummy")
outputs = model(dataset)
```

### Output

* **Classification**: `labels`, `preds`, `logits`, `time_taken`, `is_logit`.
* **Regression**: `targets`, `preds`, `time_taken`.

### Available Models
* Dummy ("dummy"): For testing and prototyping.
* OLVF ("olvf"): Supports binary and multi-class classification.
* OLIFL ("olifl"): Supports binary and multi-class classification.

---
## Versions

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

Haphazard is designed for **easy extensibility**. You can add new datasets and models.

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

    def fit(self, dataset: BaseDataset, mask_params=None, model_params=None, seed=42):
        # Dummy implementation
        x, y = dataset.load_data()
        mask = dataset.load_mask(**mask_params)
        preds = np.random.randint(0, 2, size=y.shape[0])
        if dataset.task == "classification":
            return {
                "labels": y,
                "preds": preds,
                "logits": preds.astype(float),
                "time_taken": 0.0,
                "is_logit": True
            }
        elif dataset.task == "regression":
            return {
                "targets": y,
                "preds": preds,
                "time_taken": 0.0,
            }
```

3. The model is automatically registered and can be loaded with `load_model("my_model")`.

---

## License

MIT License.
