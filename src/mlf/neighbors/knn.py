import numpy as np
from ..base import BaseEstimator

class KNN(BaseEstimator):
    """
    K-Nearest Neighbors from scratch

    Parameters:
    k: int (number of neighbors)
    task: str (classification or regresion)
    """

    def __init__(self, k=3, task="classification"):
        self.k = k
        self.task = task
        if task not in ("classification", "regression"):
            raise ValueError("Invalid task")

    def fit(self, X, y):
        self.X_train = np.asarray(X)
        self.y_train = np.asarray(y)
        return self
    
    def _euclidean_distance(self, x):
        return np.sqrt(((self.X_train - x) ** 2).sum(axis=1))
    
    def predict(self, X):
        X = np.asarray(X)
        preds = []

        for x in X:
            distance = self._euclidean_distance(x)

            # get indices of k smallest distances
            k_idx = np.argsort(distance)[:self.k]

            k_labels = self.y_train[k_idx]

            # classification or regression
            if self.task == "classification":
                values, counts = np.unique(k_labels, return_counts=True)
                pred = values[np.argmax(counts)]
            else:
                pred =  np.mean(k_labels)
            
            preds.append(pred)

        return np.array(preds)
        
        
    def score(self, X, y):
        y_pred = self.predict(X)

        if self.task == "classification":
            return np.mean(y_pred == y)
        else:
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            return 1 - ss_res / ss_tot