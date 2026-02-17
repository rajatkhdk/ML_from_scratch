import numpy as np


class DecisionTree:
    """
    Decision Tree (Classification + Regression) from scratch

    Parameters
    ----------
    max_depth : int
    min_samples_split : int
    task : "classification" or "regression"
    """

    # ----------------------------
    # Node structure
    # ----------------------------
    class Node:
        def __init__(self,
                     feature=None,
                     threshold=None,
                     left=None,
                     right=None,
                     value=None):

            self.feature = feature      # split feature index
            self.threshold = threshold  # split value
            self.left = left            # left child
            self.right = right          # right child
            self.value = value          # leaf prediction

        def is_leaf(self):
            return self.value is not None

    # ----------------------------
    # Init
    # ----------------------------
    def __init__(self, max_depth=10, min_samples_split=2, task="classification"):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.task = task
        self.root = None

    # =========================================================
    # --------------------- FIT -------------------------------
    # =========================================================
    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        self.n_features = X.shape[1]
        self.root = self._grow_tree(X, y, depth=0)

        return self

    # ----------------------------
    # Grow tree recursively
    # ----------------------------
    def _grow_tree(self, X, y, depth):

        n_samples = len(y)

        # stopping conditions
        if (depth >= self.max_depth or
            n_samples < self.min_samples_split or
            self._is_pure(y)):

            leaf_value = self._leaf_value(y)
            return self.Node(value=leaf_value)

        # find best split
        feat, thresh = self._best_split(X, y)

        if feat is None:
            return self.Node(value=self._leaf_value(y))

        # split data
        left_mask = X[:, feat] <= thresh
        right_mask = ~left_mask

        left = self._grow_tree(X[left_mask], y[left_mask], depth + 1)
        right = self._grow_tree(X[right_mask], y[right_mask], depth + 1)

        return self.Node(feat, thresh, left, right)

    # =========================================================
    # -------------------- SPLITTING ---------------------------
    # =========================================================
    def _best_split(self, X, y):

        best_gain = -1
        split_idx, split_thresh = None, None

        parent_impurity = self._impurity(y)

        for feat in range(self.n_features):

            thresholds = np.unique(X[:, feat])

            for t in thresholds:
                left_mask = X[:, feat] <= t
                right_mask = ~left_mask

                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue

                gain = self._information_gain(y, y[left_mask], y[right_mask], parent_impurity)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat
                    split_thresh = t

        return split_idx, split_thresh

    # ----------------------------
    # Information gain
    # ----------------------------
    def _information_gain(self, parent, left, right, parent_impurity):

        n = len(parent)

        child_impurity = (
            len(left)/n * self._impurity(left) +
            len(right)/n * self._impurity(right)
        )

        return parent_impurity - child_impurity

    # =========================================================
    # -------------------- IMPURITY ---------------------------
    # =========================================================
    def _impurity(self, y):
        if self.task == "classification":
            return self._gini(y)
        else:
            return self._variance(y)

    def _gini(self, y):
        _, counts = np.unique(y, return_counts=True)
        p = counts / len(y)
        return 1 - np.sum(p**2)

    def _variance(self, y):
        return np.var(y)

    # =========================================================
    # -------------------- LEAF -------------------------------
    # =========================================================
    def _leaf_value(self, y):
        if self.task == "classification":
            values, counts = np.unique(y, return_counts=True)
            return values[np.argmax(counts)]
        else:
            return np.mean(y)

    def _is_pure(self, y):
        return len(np.unique(y)) == 1

    # =========================================================
    # -------------------- PREDICT -----------------------------
    # =========================================================
    def predict(self, X):
        X = np.asarray(X)
        return np.array([self._traverse(x, self.root) for x in X])

    def _traverse(self, x, node):

        if node.is_leaf():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse(x, node.left)
        else:
            return self._traverse(x, node.right)

    # =========================================================
    # -------------------- SCORE -------------------------------
    # =========================================================
    def score(self, X, y):
        y_pred = self.predict(X)

        if self.task == "classification":
            return np.mean(y_pred == y)
        else:
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            return 1 - ss_res / ss_tot
