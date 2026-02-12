import numpy as np
from mlf.linear_model.logistic_regression import LogisticRegression

X = np.random.randn(500,2)
y = (X[:,0] + X[:,1] > 0).astype(int)

model = LogisticRegression().fit(X, y)

print("Accuracy:", model.score(X, y))