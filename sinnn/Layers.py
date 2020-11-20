import numpy as np


class ReLU:
    """Applies the Rectified Linear Unit transformation and backpropogates loss gradient"""

    def __init__(self):
        """Initialises ReLU."""

        self.input = None
        self.output = None
        pass

    def reset(self):
        """resets layer parameters"""
        self.input = None
        self.output = None

    def forward(self, x):
        """
        Applies the relu function on inputs and stores the results in outputs.

        Parameters
        ----------
        x : ndarray
            input of layer

        Notes
        -----
        output = (input if input > 0 else 0)

        """

        self.input = x
        self.output = np.maximum(self.input, 0.0)

    def backward(self, grad_output):
        """
        Backpropogates the loss gradient to calculate dL/dx.

        Relu layer does not have any weights so no need to calculate dL/dw.

        Parameters
        ----------
        grad_output : ndarray
            2d Array of same shape as `self.input` representing output gradient

        Returns
        -------
        ndarray
            2d Array of same shape as `self.input` representing input gradient

        Notes
        -----
        dL/dx = dL/dz * dz/dx
        dz/dx = (1 if input > 0 else 0)
            where:
            dL/dx = input gradient
            dL/dz = output gradient
            dz/dx = ReLU gradient

        """

        relu_grad = self.input > 0
        grad_input = grad_output * relu_grad
        return grad_input


class Dense:
    """Performs the transformation f(X) = X @ W and backpropogates loss gradient"""

    def __init__(self, neurons, weights_initialiser=None, bias_initialiser=None, optimiser=None):
        """
        Initialises layer parameters.

        Parameters
        ----------
        neurons : int
            number of neurons in layer
        weights_initialiser : object
            the weights initialiser to be used. Instance of any class from module Initialsers may be used.
        bias_initialiser : object
            the bias initialiser to be used. Instance of any class from module Initialsers may be used.
        optimiser: object
            the optimiser to be used. Instance of any class from module Optimisers may be used.
        """

        self.neurons = neurons
        self.weights_init = None
        self.biases_init = None
        self.weights = None
        self.input = None
        self.output = None
        self.weights_initialiser = weights_initialiser
        self.bias_initialiser = bias_initialiser
        self.optimiser = optimiser

    def init_weights(self, input_dim):
        """
        Initialise dense layer weights.

        Parameters
        ----------
        input_dim : int
            Number of columns in input matrix.

        """

        self.weights_init = self.weights_initialiser.initialise(input_dim, self.neurons)

    def init_biases(self):
        """Initialise dense layer biases"""

        self.biases_init = self.bias_initialiser.initialise(1, self.neurons)

    def reset(self):
        """Resets learnable layer parameters and initialises weights vector. Weights shape: [input_dim + 1, neurons]"""

        self.input = None
        self.output = None
        self.weights = np.vstack((self.biases_init, self.weights_init))
        self.optimiser.reset()

    def forward(self, x):
        """
        Calculates y = X @ W

        Parameters
        ----------
        x : ndarray
            input matrix of shape (batch_size, `input_dim` + 1)

        """

        # Initialising weights for first time use
        if self.weights is None:
            self.init_weights(x.shape[1])
            self.init_biases()
            self.reset()
        # adding a neuron/column of ones to support bias in weights matrix
        self.input = np.hstack((np.ones((x.shape[0], 1)), x))
        # self.output shape: (batch size, self.neurons)
        self.output = np.dot(self.input, self.weights)

    def backward(self, grad_output):
        """
        Backpropogates incoming gradient output to adjust layer weights and returns gradient input

        Parameters
        ----------
        grad_output : ndarray
            2d Array of shape (batch size, self.neurons)

        Returns
        -------
        ndarray
            2d Array of same shape as self.input representing input gradient

        Notes
        -----
        dL/dw = dz/dw * dL/dz
        dz/dw = x.T
        where:
            dL/dw = weights gradient
            dL/dz = output gradient
            x.T = inputs transposed
        dL/dx = dL/dz * dz/dx
        dz/dx = w.T
        where:
            dL/dx = input gradient
            dL/dz = output gradient
            w.T = weights transposed
        """
        # in calculatating grad_input, biases.T don't need to be multiplied to grad_output therefore are removed.
        grad_input = np.dot(grad_output, self.weights[1:, :].T)
        grad_weights = np.dot(self.input.T, grad_output)
        self.weights = self.optimiser.optimise(self.weights, grad_weights)
        return grad_input
