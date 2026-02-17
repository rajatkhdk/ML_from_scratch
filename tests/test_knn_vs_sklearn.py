import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from mlf.neighbors.knn import KNN

def test_knn_against_sklearn():
    # dataset
    data = load_iris()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    my = KNN(k=5).fit(X_train, y_train)
    my_acc = my.score(X_test, y_test)

    sk = KNeighborsClassifier(n_neighbors=5)
    sk.fit(X_train, y_train)
    sk_acc = sk.score(X_test, y_test)

    print("Scratch accuracy:", my_acc)
    print("Sklearn accuracy", sk_acc)

    assert abs(my_acc - sk_acc) < 0.05