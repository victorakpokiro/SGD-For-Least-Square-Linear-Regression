import numpy as np
from utils.misc import Metrics

class BaseRegression(object):
    """docstring for BaseRegression"""
    def __init__(self, optimizer, epoch=10000, log=True):
        super(BaseRegression, self).__init__()
        self.optimizer = optimizer
        self.epoch = epoch
        self.weights = None
        self.bias = None
        self.log = log



    def fit(self, X, y,batch_size=1):

        n_samples, n_features = X.shape
        self.weights = np.random.randn(n_features)
        self.bias = np.random.randn(1)
        mse = [] if self.log else None


        for _ in range(self.epoch):
            self.weights, self.bias, y_pred, y_v = self.optimizer.sgd(X,y,self.weights, self.bias, batch_size, self._approximation)
            if self.log:
                mse.append(Metrics.mse(y_pred, y_v)) #Not the best way but ok for a class project

        return mse


    def _approximation(self, X, w, b):
        raise NotImplemtedError()

    def predict(self, X):
        return self._predict(X, self.weights, self.bias)

    def _predict():
        raise NotImplemtedError()
