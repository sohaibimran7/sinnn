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

    def forward(self):
        """ Propogate input throught the layer to calculate layer outputs """
        pass

    def backward(self):
        """ Backpropogate incoming gradient to adjust layer parameters and inputs """
        pass


class ReLU(Layer):
    def __init__(self):
        """Relu layer aplies relu to inputs 1:1 fashion"""
        self.input = None
        pass

    def forward(self, input):
        """If input > 0 return input else return 0"""
        self.input = input
        return np.maximum(self.input, 0.0)

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

    def __init__(self, neurons, weights_initialiser="SmallRandom", bias_initialiser="Zeros", optimiser="SGD"):
        """Initialise layer parameters"""
        self.neurons = neurons
        self.weights_initialiser = getattr(Initialisers, weights_initialiser)()
        self.bias_initialiser = getattr(Initialisers, bias_initialiser)()
        self.weights_init = None
        self.biases_init = None
        self.weights = None
        self.input = None
        self.optimiser = getattr(Optimisers, optimiser)()

    def init_weights(self, input_dim):
        """Initialise dense layer weights. Requires shape/neurons/coloumns of input"""
        self.weights_init = self.weights_initialiser.initialise(input_dim, self.neurons)

    def init_biases(self):
        """Initialise dense layer weights. Requires shape/neurons/coloumns of input"""
        self.biases_init = self.weights_initialiser.initialise(1, self.neurons)

    def reset(self):
        """
        Resets learnable layer parameters.
        Weights shape: [input_dim + 1, neurons]
        """
        self.weights = np.vstack((self.biases_init, self.weights_init))
        self.optimiser.reset()

    def forward(self, input):
        """
        Calculates y = X @ W

        X shape: [batch, input_dim + 1]
        Y shape: [batch, neurons]
        """
        # adding a neuron/column of ones to support bias in weights matrix
        self.input = np.hstack((np.ones((input.shape[0], 1)), input))
        return np.dot(self.input, self.weights)

    def backward(self, grad_output):
        """
          Backpropogate incoming gradient output to adjust layer weights and returns gradient input
        """
        # in calculatating grad_input, biases.T don't need to be multiplied to grad_output therefore are removed.
        grad_input = np.dot(grad_output, self.weights[1:, :].T)
        grad_weights = np.dot(self.input.T, grad_output)
        self.weights = self.optimiser.optimise(self.weights, grad_weights)
        return grad_input
