import numpy as np
import Losses


def batches(x, y,  batch_size):
    """Yield successive batch_sized chunks from lst."""
    for i in range(0, len(x), batch_size):
        yield x[i:i + batch_size], y[i:i + batch_size]


class Model:

    def __init__(self, weights_initialiser="SmallRandom", bias_initialiser="Zeros",
                 loss_function="CategoricalCrossEntropy", optimiser="SGD"):
        self.weights_initialiser = weights_initialiser
        self.bias_initialiser = bias_initialiser
        self.loss_function = getattr(Losses, loss_function)()
        self.optimiser = optimiser
        self.layers = []
        self.input = None
        pass

    def add(self, *layers):
        for layer in layers:
            if hasattr(layer, 'weights_initialiser'):
                if not layer.weights_initialiser:
                    layer.weights_initialiser = self.weights_initialiser
            if hasattr(layer, 'bias_initialiser'):
                if not layer.bias_initialiser:
                    layer.bias_initialiser = self.bias_initialiser
            if hasattr(layer, 'optimiser'):
                if not layer.optimiser:
                    layer.optimiser = self.optimiser
            self.layers.append(layer)

    def reset(self):
        self.input = None
        self.loss_function.reset()
        map(lambda layer: layer.reset(), self.layers)

    def forward(self, x):
        self.input = x
        for index, layer in enumerate(self.layers):
            layer.forward(self.layers[index - 1].output) if index > 0 else layer.forward(self.input)

    def predict(self, x):
        self.forward(x)
        return self.layers[-1].output

    def train(self, x_train, y_train, batch_size, epochs, x_val, y_val):
        assert len(x_train) == len(y_train), "x_train and y_train must have the same length"
        assert len(x_val) == len(y_val), "x_val and y_val must have the same length"
        assert len(x_train) % batch_size == 0, "training set must be divisible by batch_size"
        for i in range(epochs):
            for x_batch, y_batch in batches(x_train, y_train, batch_size):
                loss_gradient = self.loss_function.loss_gradient(self.predict(x_batch), y_batch)

                for index, layer in enumerate(reversed(self.layers)):
                    loss_gradient = layer.backward(loss_gradient)

            train_accuracy = (np.argmax(self.predict(x_train), axis=1) == y_train).mean()
            validation_accuracy = (np.argmax(self.predict(x_val), axis=1) == y_val).mean()
            print("epochs: {}\ttrain_accuracy: {}\tvalidation_accuracy: {}".format(i + 1, train_accuracy, validation_accuracy))
