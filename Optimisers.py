class Optimiser:
    """ Stores common optimiser functions """
    def __init__(self):
        """ Initialise optimiser parameters"""
        pass

    def reset(self):
        """ Reset optimiser parameters"""
        pass

    def optimise(self, weights, grad_weights):
        """ Optimise using parameters"""
        pass


class SGD(Optimiser):
    """ Implements stochastic gradient descent. """
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def reset(self):
        pass

    def optimise(self, weights, grad_weights):
        return weights - self.learning_rate * grad_weights


class Momentum(Optimiser):
    """ Implements stochastic gradient descent with Momentum. """
    def __init__(self, learning_rate=0.01, eta=0.9):
        self.learning_rate = learning_rate
        self.eta = eta
        self.momentum = None

    def reset(self):
        self.momentum = 0

    def optimise(self, weights, grad_weights):
        self.momentum = self.eta * self.momentum + self.learning_rate * grad_weights
        return weights - self.momentum
