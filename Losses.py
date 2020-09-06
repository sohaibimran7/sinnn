import numpy as np


def softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=1, keepdims=True)


def one_hot(labels, categories):
    y = np.zeros((labels.size, categories))
    y[np.arange(labels.size), labels] = 1
    return y


class Loss():
    def __init__(self):
        pass

    def reset(self):
        pass

    def loss_gradient(self, x, y):
        pass


class CategoricalCrossEntropy(Loss):

    def __init__(self, calc_loss=False):
        self.loss = None
        self.calc_loss = calc_loss
        pass

    def reset(self):
        self.loss = None

    def loss_gradient(self, x, y):
        p = softmax(x)

        one_hot_y = one_hot(y, x.shape[1])

        if self.calc_loss:
            self.loss = np.mean(np.multiply(one_hot_y, np.log(p)))

        return (p - one_hot_y) / len(y)
