import numpy as np

class SGD(object):
    """docstring for SGD"""
    def __init__(self, lr=0.001):
        super(SGD, self).__init__()
        self.lr = lr


    def sgd( self ,X,y, weights, bias, batch_size, func):

        indexes = np.random.randint(0, len(X), batch_size)
        Xs = X[indexes]

        #ys = np.take(y, indexes)
        ys = y[indexes]
        N = len(Xs)
        y_predicted = func(Xs, weights, bias)
        dw = (1/N) * 2* np.dot(Xs.T, (y_predicted - ys))
        db = (1/N) *2* np.sum(y_predicted-ys)

        weights -= self.lr * dw
        bias -= self.lr * db
        y_predicted = func(Xs, weights, bias)

        return weights, bias , y_predicted, ys
