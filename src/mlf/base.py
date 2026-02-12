class BaseEstimator:
    def fit(self, X, y):
        raise NotImplementedError
    
    def predict(self, X):
        raise NotImplementedError
    
    def score(self, X, y):
        preds = self.predict(X)
        return (preds == y).mean()