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
        #exit()

        for _ in range(self.epoch):
            #y_predicted = np.dot(X, self.weights) + self.bias
            # indexes = np.random.randint(0, len(X), batch_size)
            # Xs = np.take(X, indexes)
            # ys = np.take(y, indexes)
            # N = len(Xs)
            # y_predicted = self._approximation(Xs, self.weights, self.bias)
            # dw = (1/N) * 2* np.dot(Xs.T, (y_predicted - ys))
            # db = (1/N) *2* np.sum(y_predicted-ys)

            # self.weights -= self.lr * dw
            # self.bias -= self.lr * db
            self.weights, self.bias = self.optimizer.sgd(X,y,self.weights, self.bias, batch_size, self._approximation)


    def _approximation(self, X, w, b):
        raise NotImplemtedError()

    def predict(self, X):
        return self._predict(X, self.weights, self.bias)

    def _predict():
        raise NotImplemtedError()
