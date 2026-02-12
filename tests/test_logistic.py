import numpy as np
from mlf.linear_model.logistic_regression import LogisticRegression

def test_logic():
    X = np.random.randn(200,2)
    y = (X[:,0] > 0).astype(int)

    model = LogisticRegression().fit(X, y)
    acc = model.score(X, y)

    assert acc > 0.9