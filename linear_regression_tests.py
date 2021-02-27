import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from models.linear_regression import LinearRegression

def mse(y_true, y_predicted):
    return np.mean((y_true - y_predicted)**2)


def init_data():
    X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    regressor = LinearRegression(lr=0.01, n_iters=100000)
    regressor.fit(X_train, y_train)
    predicted = regressor.predict(X_test)
    mse_value = mse(y_test, predicted)
    print(mse_value)
    y_pred_line = regressor.predict(X)
    cmap = plt.get_cmap('viridis')
    fig = plt.figure(figsize=(8,6))
    m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s=10)
    m2 = plt.scatter(X_test, y_test, color=cmap(0.5), s=10)
    plt.plot(X, y_pred_line, color = "black", linewidth=2, label="Predicted")
    plt.show()

def main():
    init_data()

if __name__ == '__main__':
    main()
