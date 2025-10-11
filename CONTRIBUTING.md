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
        self.task = "classification"  # or "regression"
        super().__init__(base_path=base_path, **kwargs)

    def read_data(self, base_path="./"):
        x = np.random.random((100, 10))
        y = np.random.randint(0, 2, 100)
        return x, y
```

3. **Test loading your dataset**:

```python
from haphazard import load_dataset
dataset = load_dataset("my_dataset")
x, y = dataset.load_data()
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

    def fit(self, dataset: BaseDataset, mask_params=None, model_params=None, seed=42):
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

3. **Test loading your model**:

```python
from haphazard import load_model
model = load_model("my_model")
results = model(dataset)
```

**Notes:**

* Use the `@register_model("my_model")` decorator for registration.
* Implement `fit()` to return the required output dictionary depending on `dataset.task`.

---

## Testing

* Write small scripts to **load datasets and models**.
* Validate that outputs conform to `BaseDataset` and `BaseModel` specifications.
* Check shapes, types, and required keys (`labels`, `preds`, `logits`, `is_logit`, etc.).

---

## Pull Requests

1. Fork the repo and create a new branch for your feature/fix.
2. Ensure your code passes basic tests.
3. Add clear docstrings and comments.
4. Open a pull request against `main`.
5. Include a description of what your PR changes and any instructions for testing.

---

## Code of Conduct

* Be respectful and collaborative.
* Follow PEP8 and maintain consistency.
* Contributions are welcome from everyone.
