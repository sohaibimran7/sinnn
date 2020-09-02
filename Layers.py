import numpy as np
import Initialisers
import Optimisers

class Layer:
    """
    This layer doesn't do much. Extend its functionality using inheritence.
    """

    def __init__(self):
        """ Initialise layer parameters """
        pass

    def forward(self, X):
        """ Propogate input throught the layer to calculate layer outputs """
        pass

    def backward(self):
        """ Backpropogate incoming gradient to adjust layer parameters and inputs """
        pass


class ReLU(Layer):
    def __init__(self, **kwargs):
        """Relu layer aplies relu to inputs 1:1 fashion"""
        self.input = None
        self.output = None
        pass

    def reset(self):
        self.input = None
        self.output = None

    def forward(self, X):
        """If input > 0 return input else return 0"""
        self.input = X
        self.output = np.maximum(self.input, 0.0)
        return self.output

    def backward(self, grad_output):
        """compute DL/DX only since no weights"""
        """DL/DX = DL/DY * DY/DX"""
        """if input > 0 dY/dX = 1 else dY/dX = 0"""
        relu_grad = self.input > 0
        return grad_output * relu_grad


class Dense(Layer):
    """
    Performs the transformation f(X) = X @ W and backprops
    """

    def __init__(self, neurons, **kwargs):
        """Initialise layer parameters"""
        self.neurons = neurons
        self.weights_init = None
        self.biases_init = None
        self.weights = None
        self.input = None
        self.output = None
        self.weights_initialiser = None
        self.bias_initialiser = None
        self.optimiser = None
        allowed_keys = {'weights_initialiser', 'bias_initialiser', 'optimiser'}
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in allowed_keys)

    def objectify(self):
        self.weights_initialiser = getattr(Initialisers, self.weights_initialiser)()
        self.bias_initialiser = getattr(Initialisers, self.bias_initialiser)()
        self.optimiser = getattr(Optimisers, self.optimiser)()

    def init_weights(self, input_dim):
        """Initialise dense layer weights. Requires shape/neurons/coloumns of input"""
        self.weights_init = self.weights_initialiser.initialise(input_dim, self.neurons)

    def init_biases(self):
        """Initialise dense layer weights. Requires shape/neurons/coloumns of input"""
        self.biases_init = self.bias_initialiser.initialise(1, self.neurons)

    def reset(self):
        """
        Resets learnable layer parameters.
        Weights shape: [input_dim + 1, neurons]
        """
        self.input = None
        self.output = None
        self.weights = np.vstack((self.biases_init, self.weights_init))
        self.optimiser.reset()

    def forward(self, X):
        """
        Calculates y = X @ W

        X shape: [batch, input_dim + 1]
        Y shape: [batch, neurons]
        """
        # Initialising weights for first time use
        if not self.weights:
            self.objectify()
            self.init_weights(X.shape[1])
            self.init_biases()
            self.reset()
        # adding a neuron/column of ones to support bias in weights matrix
        self.input = np.hstack((np.ones((X.shape[0], 1)), input))
        self.output = np.dot(self.input, self.weights)
        return self.output

    def backward(self, grad_output):
        """
          Backpropogate incoming gradient output to adjust layer weights and returns gradient input
        """
        # in calculatating grad_input, biases.T don't need to be multiplied to grad_output therefore are removed.
        grad_input = np.dot(grad_output, self.weights[1:, :].T)
        grad_weights = np.dot(self.input.T, grad_output)
        self.weights = self.optimiser.optimise(self.weights, grad_weights)
        return grad_input
