import numpy as np
from sklearn.linear_model import LogisticRegression as SKLogistic
from mlf.linear_model.logistic_regression import LogisticRegression

def test_against_sklearn():
    np.random.seed(0)

    X = np.random.randn(1000, 5)
    y = (X[:,0] + 0.5*X[:,1] > 0).astype(int)

    my = LogisticRegression(lr=0.1, epochs=2000).fit(X, y)
    sk = SKLogistic(max_iter=2000).fit(X,y)

    my_acc = my.score(X, y)
    sk_acc = sk.score(X, y)

    print("My:", my_acc)
    print("SK:",sk_acc)

    print("My weights:", my.w)
    print("SK weights:", sk.coef_)

    # assert np.allclose(my.w, sk.coef_, atol=1e-1)

    assert abs(my_acc - sk_acc) < 0.02