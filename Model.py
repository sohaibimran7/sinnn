class Model:

    def __init__(self, weights_initialiser="SmallRandom", bias_initialiser="Zeros", optimiser="SGD"):
        self.weights_initialiser = weights_initialiser
        self.bias_initialiser = bias_initialiser
        self.optimiser = optimiser
        self.layers = None
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


    def forward(self, X):
        self.input = X
        for index,layer in enumerate(self.layers):
            layer.forward(self.layers[index - 1].output) if index > 0 else layer.forward(self.input)

    def predict(self, X):
        self.forward(X)
        return self.layers[-1].output


    def train(self, X, Y):
        pass
