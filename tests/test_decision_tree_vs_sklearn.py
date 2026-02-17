import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from mlf.tree.decision_tree import DecisionTree

def test_knn_against_sklearn():
    # dataset
    X, y = load_iris(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # your tree
    my_tree = DecisionTree(max_depth=4, task="classification")
    my_tree.fit(X_train, y_train)

    # sklearn tree
    sk_tree = DecisionTreeClassifier(max_depth=4)
    sk_tree.fit(X_train, y_train)

    my_tree_score = my_tree.score(X_test, y_test)
    sk_tree_score = sk_tree.score(X_test, y_test)

    print("My accuracy:", my_tree_score)
    print("Sklearn accuracy:", sk_tree_score)

    assert abs(my_tree_score - sk_tree_score) < 0.05
