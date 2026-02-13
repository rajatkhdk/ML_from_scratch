from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression as SKLinear
from mlf.linear_model.linear_regression import LinearRegeression

def test_against_sklearn():
    data = load_diabetes()
    X,y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    my = LinearRegeression(lr=0.01, epochs=100).fit(X_train, y_train)
    sk = SKLinear().fit(X_train, y_train)

    my_score = my.score(X_test, y_test)
    sklearn_score = sk.score(X_test, y_test)

    print("Scratch R2:", my_score)
    print("SKLearn R2:", sklearn_score)

    assert abs(sklearn_score - my_score) < 0.5