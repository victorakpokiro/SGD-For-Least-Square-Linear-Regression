import numpy as np
from .base_regressor import BaseRegression

class LinearRegression(BaseRegression):
    """docstring for LinearRegression"""
    def __init__(self, lr=0.001, n_iters=1000):
        super(LinearRegression, self).__init__(lr, n_iters)

    def _predict(self, X, weights, bias):
        return np.dot(X, weights) + bias

    def _approximation(self, X, weights, bias):
        return np.dot(X, weights) + bias
