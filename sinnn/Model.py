import numpy as np
from copy import deepcopy

from sinnn.Initialisers import SmallRandom, Zeros
from sinnn.Losses import CrossEntropy
from sinnn.Optimisers import AdaGrad
from sinnn.utils import batches, sigmoid
from sinnn import Metrics


class Model:
    """The neural network model"""

    def __init__(self, weights_initialiser=SmallRandom(), bias_initialiser=Zeros(),
                 loss_function=CrossEntropy(), optimiser=AdaGrad()):
        """
        Initialises Model parameters.

        Parameters
        ----------
        weights_initialiser : object
            the weights initialiser to be used. Instance of any class from module Initialsers may be used.
        bias_initialiser : object
            the bias initialiser to be used. Instance of any class from module Initialsers may be used.
        loss_function : object
            the loss function to be used. Instance of any class from module Losses may be used.
        optimiser: object
            the optimiser to be used. Instance of any class from module Optimisers may be used.

        """

        self.weights_initialiser = weights_initialiser
        self.bias_initialiser = bias_initialiser
        self.loss_function = loss_function
        self.optimiser = optimiser
        self.isbinary = None
        self.layers = []
        self.train_log = {}
        pass

    def add(self, *layers):
        """
        Adds a layer to the model

        Layers are stored in model.layers. Model parameters are passed to every class that doesn't already have those
        parameters defined at the level of the layer. This allows complete flexibility in setting layers with different
        parameters to each other.

        Parameters
        ----------
        layers : object
            the layer to be added. Instance of any class from module Layers may be used.

        """

        for layer in layers:
            if hasattr(layer, 'weights_initialiser'):
                if not layer.weights_initialiser:
                    layer.weights_initialiser = deepcopy(self.weights_initialiser)
            if hasattr(layer, 'bias_initialiser'):
                if not layer.bias_initialiser:
                    layer.bias_initialiser = deepcopy(self.bias_initialiser)
            if hasattr(layer, 'optimiser'):
                if not layer.optimiser:
                    layer.optimiser = deepcopy(self.optimiser)
            self.layers.append(layer)

    def reset(self):
        """Calls the reset method for all layers in model"""
        map(lambda layer: layer.reset(), self.layers)

    def logits(self, x):
        """
        Completes a forward pass through all the layers of the model to calculate logits.

        Parameters
        ----------
        x : ndarray
            2d input array. The first layer's input contain input of the model, while the latter layer's inputs are
            their previous layers outputs.

        Returns
        -------
        ndarray
            array of logits.

        """
        for index, layer in enumerate(self.layers):
            layer.forward(self.layers[index - 1].output) if index > 0 else layer.forward(x)
        return self.layers[-1].output

    def predict(self, x):
        """
        Completes a forward pass through all the layers of the model to calculate predictions.

        Parameters
        ----------
        x : ndarray
            2d input array. The first layer's input contain input of the model, while the latter layer's inputs are
            their previous layers outputs.

        Returns
        -------
        ndarray
            1d array of predicted labels.
        """
        return (np.round(sigmoid(self.logits(x)))
                if self.isbinary else
                np.argmax(self.logits(x), axis=1)) \
            if self.loss_function.__class__.__name__ == 'CrossEntropy' \
            else self.logits(x)

    def train(self, x_train, y_train, batch_size, epochs, validation=(), metrics=("loss",)):
        """
        Trains the model.

        Parameters
        ----------
        x_train: ndarray
            2d array containing train data of shape (number of examples, predictors per example)
        y_train: ndarray
            1d or 2d array containing train labels of shape (number of examples,) or (number of examples, 1)
        batch_size: int
            defines how many examples each batch should contain
        epochs : int
            defines how many epochs should the training include
        validation : tuple
            tuple of ndarrays of len 2 containing the validation inputs and labels.
        metrics : tuple
            tuple of strings defining what metrics should be computed by the model. Any function from module Metrics may
            be given.

        """

        #Exception handling
        if not len(x_train) == len(y_train):
            raise Exception("x_train and y_train must have the same length")
        if not len(x_train) % batch_size == 0:
            raise Exception("training set must be divisible by batch_size")

        # check if validation data has been provided.
        if len(validation) > 0:
            if not len(validation) == 2:
                raise Exception(f"Validation accepts exactly 2 arguments(x_val, y_val). {len(validation)} were given")
            if not len(validation[0]) == len(validation[1]):
                raise Exception("x_val and y_val must have the same length")
            else:
                x_val, y_val = validation[0], validation[1]
                val = True
        else:
            x_val, y_val = None, None
            val = False

        self.isbinary = (True if self.layers[-1].neurons == 1 else False)

        # If labels as 1d array, reshape to 2d array
        if self.isbinary:
            if self.loss_function.isbinary is None:
                self.loss_function.isbinary = True
            if y_train.ndim == 1:
                y_train = y_train.reshape(-1, 1)
            if val:
                if y_val.ndim == 1:
                    y_val = y_val.reshape(-1, 1)

        # Metrics reporting
        print(self.report_metrics(0, x_train, y_train, x_val, y_val, val, metrics))

        # training
        for i in range(epochs):

            # splitting into batches for training
            for x_batch, y_batch in batches(x_train, y_train, batch_size):
                loss_gradient = self.loss_function.loss_gradient(self.logits(x_batch), y_batch)

                for index, layer in enumerate(reversed(self.layers)):
                    loss_gradient = layer.backward(loss_gradient)

            # Metrics reporting
            print(self.report_metrics(i+1, x_train, y_train, x_val, y_val, val, metrics))

    def report_metrics(self, epoch, x_train, y_train, x_val, y_val, val, metrics):
        """
        Computes and reports metrics.

        Parameters
        ----------
        epoch : int
            which epoch is currently running
        x_train: ndarray
            2d array containing train data of shape (number of examples, predictors per example)
        y_train: ndarray
            1d or 2d array containing train labels of shape (number of examples,) or (number of examples, 1)
        x_val: ndarray
            2d array containing validation data of shape (number of examples, predictors per example)
        y_val: ndarray
            1d or 2d array containing validation labels of shape (number of examples,) or (number of examples, 1)
        val
        metrics

        Returns
        -------
        dict

        """
        m = dict()
        m["Epochs"] = epoch
        if epoch == 0 and self.train_log:
            self.train_log = {}

        logits = ([self.logits(x_train), self.logits(x_val)] if val else [self.logits(x_train)])
        labels = ([y_train, y_val] if val else [y_train])

        if type(metrics) == str:
            metrics = (metrics,)

        for metric in metrics:
            if metric not in self.train_log:
                self.train_log[metric] = {}
            keys = ([f"train_{metric}", f"validation_{metric}"] if val else [f"train_{metric}"])

            for i, key in enumerate(keys):
                if key not in self.train_log[metric]:
                    self.train_log[metric][key] = []
                met = getattr(Metrics, metric)(logits[i], labels[i], self.loss_function.__class__.__name__, self.isbinary)
                m[key] = met
                self.train_log[metric][key].append(met)

        return m
