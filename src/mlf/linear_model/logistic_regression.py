import numpy as np
from ..base import BaseEstimator

class LogisticRegression(BaseEstimator):
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0

        for _ in range(self.epochs):
            z = X @ self.w + self.b
            p = self._sigmoid(z)

            dw = X.T @ (p - y) / n
            db = np.mean(p-y)

            self.w -= self.lr * dw
            self.b -= self.lr * db

        return self
    
    def predict(self, X):
        probs = self._sigmoid(X @ self.w + self.b)
        return (probs > 0.5).astype(int)