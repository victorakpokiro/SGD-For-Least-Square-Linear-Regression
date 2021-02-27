import numpy as np
from .base_regressor import BaseRegression

class LinearRegression(BaseRegression):
    """docstring for LinearRegression"""
    def __init__(self, optimizer, epoch=1000):
        super(LinearRegression, self).__init__(optimizer, epoch)

    def _predict(self, X, weights, bias):
        return np.dot(X, weights) + bias

    def _approximation(self, X, weights, bias):
        #print(X.shape, weights.shape, bias)
        return np.dot(X, weights) + bias
