import numpy as np

class SGD:
    """Implements stochastic gradient descent."""

    def __init__(self, learning_rate=0.01):
        """
        Initialises SGD parameters.

        Parameters
        ----------
        learning_rate : float
            learning rate of the optimiser
        """
        self.learning_rate = learning_rate

    def reset(self):
        pass

    def optimise(self, weights, grad_weights):
        """
        Moves the weights in the direction opposite to weights gradient to attempt to reduce the Loss.

        Stochastic gradient descent is used to make small adjustments to the weights in an attempt to minimize the Loss.
        The size of the adjustments are determined by the learning rate.

        Parameters
        ----------
        weights : ndarray
            The existing weights which the optimiser needs to change.
        grad_weights : ndarray
            The weights gradient that determine which direction do the `weights` move

        Returns
        -------
        ndarray
            The new weights

        Notes
        -----
        w1 = w0 - n(dL/dw)
        where:
            w1 = the new weights
            w0 = The existing weights
            n = learning rate
            dL/dw = The weights gradient

        """
        return weights - self.learning_rate * grad_weights


class Momentum:
    """ Implements stochastic gradient descent with Momentum. """

    def __init__(self, learning_rate=0.01, eta=0.9):
        """
        Initialises Momentum parameters.

        Parameters
        ----------
        learning_rate : float
            learning rate of the optimiser
        eta : float
            momentum multiplier

        """

        self.learning_rate = learning_rate
        self.eta = eta
        # Momentum vector initialised to None
        self.momentum = None

    def reset(self):
        """Resets momentum to 0"""
        self.momentum = 0

    def optimise(self, weights, grad_weights):
        """
        Moves the weights in the direction opposite to weights gradient and momentum to attempt to reduce the Loss.

        Momentum is an extension of Stochastic gradient descent. A momentum vector is kept in memory
        which also determines the direction of adjustments to weights, other than weight gradients.
        The momentum vector is determined by weight gradients and previous momentum.

        Parameters
        ----------
        weights : ndarray
            The existing weights which the optimiser needs to change.
        grad_weights : ndarray
            The weights gradient that determine which direction do the `weights` move

        Returns
        -------
        ndarray
            The new weights

        Notes
        -----
        h1 = a(h0) + n(dL/dw)
        w1 = w0 - h1

        where:
            w1 = the new weights
            w0 = The existing weights
            n = learning rate
            dL/dw = The weights gradient
            h1 = the new momentum
            h0 = the previous momentum

        """

        self.momentum = self.eta * self.momentum + self.learning_rate * grad_weights
        return weights - self.momentum


class AdaGrad:
    """
    Initialises AdaGrad parameters.

    Parameters
    ----------
    learning_rate : float
        learning rate of the optimiser

    """

    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        # g vector initialised to None
        self.g = None

    def reset(self):
        """Resets g to 0"""
        self.g = 0

    def optimise(self, weights, grad_weights):
        """
         Moves the weights in the direction opposite to weights gradient to attempt to reduce the Loss.

         AdaGrad is an extension of Stochastic gradient descent. AdaGrad allows for an adaptive learning rate by keeping
         a vector g that accumulates the square of grad_weights. As the grad_weights get smaller, so does the learning
         rate

         Parameters
         ----------
         weights : ndarray
             The existing weights which the optimiser needs to change.
         grad_weights : ndarray
             The weights gradient that determine which direction do the `weights` move

         Returns
         -------
         ndarray
             The new weights

         Notes
         -----
         g1 = g0 + (dL/dw)^2
         w1 = w0 - n(dL/dw)/(g1 + e)
         where:
             w1 = the new weights
             w0 = The existing weights
             g1 = the new g
             g0 = the existing g
             n = learning rate
             dL/dw = The weights gradient
             e = epsilon

         eplison is added to prevent division by 0

         """
        self.g = self.g + np.square(grad_weights)
        return weights - (self.learning_rate * grad_weights) / np.sqrt(self.g + 1E-07)
