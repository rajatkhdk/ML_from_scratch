# 📘 ML From Scratch

## This project is the implementation of machine learning algorithm using numpy.

A pure NumPy implementation of Machine Learning algorithms built **from first principles**, without using scikit-learn internally.

The goal of this repository is simple:

> If you can implement it yourself, you truly understand it.

This project focuses on mathematics, gradients, optimization, and clean engineering practices.

---

# 🎯 Objectives

- Implement classical ML algorithms manually
- Understand the math behind each model
- Derive gradients step-by-step
- Learn optimization (Gradient Descent)
- Compare results with sklearn benchmarks
- Write clean, testable, modular code

---

# 📂 Project Structure
```
ML_from_scratch/
│
├── src/mlf/
│ ├── linear_model/
│ │ ├── linear_regression.py
│ │ ├── logistic_regression.py
│ │ └── README.md
│ │
│ ├── neighbors/
│ │ ├── knn.py
│ │ └── README.md
│ │
│ ├── tree/
│ │ ├── decision_tree.py
│ │ └── README.md
│
├── examples/
├── tests/
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

# ✅ Implemented Algorithms

## Linear Models
- Linear Regression (Gradient Descent)
- Logistic Regression (Binary Classification)

## Neighbors
- K-Nearest Neighbors (KNN)

## Trees
- Decision Tree (basic implementation)

More coming soon:
- Naive Bayes
- SVM
- PCA
- Neural Networks

---

# ⚙️ Installation

## Using uv (recommended)

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
uv add -r requirement.txt
```

<!-- # Using pip
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
``` -->

# 🧪 Running Tests
All models are validated against sklearn implementations
```bash
pytest
```

# 🔬 Training Pipeline (common to all models)
- Initialize weights
- Forward pass
- Compute loss
- Compute gradients manually
- Update weights using Gradient Descent
- Repeat for epochs

### Example update
```py
w = w - lr * dw
b = b - lr * db
```

# Why This project?
Most libraries hide:
- gradient math
- optimization details
- solver behavior

This repo exposes everything.

After building models from scratch, using sklearn becomes much easier.

# 🤝 Contributing
Contributions welcome:
- add new algorithms
- improve math explanations
- add tests
- optimize iplementations