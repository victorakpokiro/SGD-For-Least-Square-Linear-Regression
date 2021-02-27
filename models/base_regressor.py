import numpy as np

class BaseRegression(object):
    """docstring for BaseRegression"""
    def __init__(self, lr=0.001, n_iters=1000):
        super(BaseRegression, self).__init__()
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):

        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            #y_predicted = np.dot(X, self.weights) + self.bias
            y_predicted = self._approximation(X, self.weights, self.bias)
            dw = (1/n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1/n_samples) * np.sum(y_predicted-y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def _approximation(self, X, w, b):
        raise NotImplemtedError()

    def predict(self, X):
        return self._predict(X, self.weights, self.bias)

    def _predict():
        raise NotImplemtedError()
