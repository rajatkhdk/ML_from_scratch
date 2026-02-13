import numpy as np
from ..base import BaseEstimator

class LinearRegeression:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.w = None
        self.b = None

    def fit(self, X, y):
        n, d = X.shape

        # initialize
        self.w = np.zeros(d)
        self.b = 0.0

        for _ in range(self.epochs):
            y_hat = X @ self.w + self.b
            error = y_hat - y

            # gradients
            dw = (2/n) * X.T @ error
            db = (2/n) * np.sum(error)

            # update
            self.w -= self.lr * dw
            self.b -= self.lr * db

        return self
    
    # prediction
    def predict(self, X):
        return X @ self.w + self.b
    
    # score (R2)
    def score(self, X, y):
        y_hat = self.predict(X)

        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        return 1 - ss_res/ss_tot
