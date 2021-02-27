import numpy as np

class BaseRegression(object):
    """docstring for BaseRegression"""
    def __init__(self, optimizer, epoch=10000):
        super(BaseRegression, self).__init__()
        self.optimizer = optimizer
        self.epoch = epoch
        self.weights = None
        self.bias = None


    def fit(self, X, y,batch_size=1):

        n_samples, n_features = X.shape
        self.weights = np.random.randn(n_features)
        self.bias = np.random.randn(1)


        for _ in range(self.epoch):
            self.weights, self.bias = self.optimizer.sgd(X,y,self.weights, self.bias, batch_size, self._approximation)


    def _approximation(self, X, w, b):
        raise NotImplemtedError()

    def predict(self, X):
        return self._predict(X, self.weights, self.bias)

    def _predict():
        raise NotImplemtedError()
