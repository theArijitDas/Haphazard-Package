# Contributing to Haphazard

Thank you for your interest in contributing to **Haphazard**! This guide explains how to add **new datasets**, **new models**, and follow best practices for code, testing, and pull requests.

---

## Table of Contents

- [Getting Started](#getting-started)
- [Coding Guidelines](#coding-guidelines)
- [Adding a New Dataset](#adding-a-new-dataset)
- [Adding a New Model](#adding-a-new-model)
- [Testing](#testing)
- [Pull Requests](#pull-requests)
- [Code of Conduct](#code-of-conduct)

---

## Getting Started

1. **Fork the repository**:

```bash
git clone https://github.com/theArijitDas/Haphazard-Package
cd haphazard
````

2. **Set up a virtual environment**:

```bash
python -m venv .venv
source .venv/bin/activate   # Linux / MacOS
.venv\Scripts\activate      # Windows
```

3. **Install dependencies for development**:

```bash
pip install -e .
```

4. **Clean pycache (optional)**:

```bash
find . -type d -name "__pycache__" -exec rm -r {} +
```

---

## Coding Guidelines

* Use **snake_case** for functions and variables.
* Use **PascalCase** for class names.
* Include **NumPy-style docstrings** for all classes and functions.
* Keep **type hints** consistent.
* Avoid global variables; use class attributes or instance variables.
* Ensure that datasets and models use the **registration decorators** (`@register_dataset` or `@register_model`) for automatic discovery.

---

## Adding a New Dataset

1. **Create a folder** under `haphazard/data/datasets/`:

```
haphazard/data/datasets/my_dataset/
```

2. **Add `__init__.py`** with your dataset:

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


3. **Test loading your dataset**:

```python
from haphazard import load_dataset
dataset = load_dataset("my_dataset")
x, y = dataset.x, dataset.y
```

**Notes:**

* `name` must be unique and case-insensitive.
* `task` must be `"classification"` or `"regression"`.
* Implement `read_data` to return `(x, y)`.

---

## Adding a New Model

1. **Create a folder** under `haphazard/models/model_zoo/`:

```
haphazard/models/model_zoo/my_model/
```

2. **Add `__init__.py`** with your model:

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


3. **Test loading your model**:

```python
from haphazard import load_model
model = load_model("my_model", model_params)
results = model(dataset)
```

**Notes:**

* Use the `@register_model("my_model")` decorator for registration.
* Implement `fit()` to return the required output dictionary depending on `dataset.task`.

---

## Adding a New Normalizer

The normalization module (introduced in v1.1.0) allows online or offline scaling and transformation of data streams.

1. **Create a folder** under:

   ```
   haphazard/normalization/normalizer_zoo/my_normalizer/
   ```

2. **Add `__init__.py`**:

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

3. **Test your normalizer**

   ```python
   from haphazard import load_normalizer
   X = np.random.randn(10, 5)
   n = load_normalizer("my_normalizer", num_features=X.shape[1])
   for x in X:
      mask = np.ones_like(x, dtype=bool)
      x_norm = n(x, mask)
      print(x_norm)
   ```

4. **Notes**

   * Always subclass `OnlineNormalization`.
   * Register with `@register_normalizer("unique_name")`.
   * Keep transformations numerically stable (add small epsilons when dividing).
   * Ensure your implementation handles streaming data.



---

## Testing

* Run minimal tests for your component:

  * Load and use dataset/model/normalizer without import errors.
  * Check output shapes, keys, and dtype consistency.
  * Validate normalization correctness with synthetic data.

---

## Pull Requests

1. Fork the repo and work on a feature branch.
2. Use descriptive commit messages.
3. Verify tests and type hints pass cleanly.
4. Open a PR against `main` with a summary of:
    * What was added/modified.
    * Any dependencies introduced.

---

## Code of Conduct

* Be respectful and collaborative.
* Follow PEP8 and maintain consistency.
* Contributions are welcome from everyone.
* All contributions, whether datasets, models, or normalizers, are equally valued.
