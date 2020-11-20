import numpy as np
from sinnn.utils import sigmoid, softmax, one_hot


class CrossEntropy:
    """Calculates the rate of change of loss with respect to logits.

    Attributes
    ----------
    isbinary : bool
        Whether CrossEntropy should calculate for binary or categorical CrossEntropy.

    """

    def __init__(self, isbinary=None):
        """
        Initialises Crossentropy.

        Parameters
        ----------
        isbinary : bool
            Whether the loss function should calculate gradient of Binary CrossEntropy or Categorical CrossEntropy.
            Initialised to None by default.
        """

        self.isbinary = isbinary
        pass

    def loss_gradient(self, z, y):
        """
        Calculates dL/dz where L represents Loss and z represents logits

        Parameters
        ----------
        z : ndarray
            Array containing logits of the current batch.
        y : ndArray
            Array containing labels of the current batch

        Returns
        -------
        ndarray
            Array of same shape as z and y containing dL/dz.

        Notes
        -----

        Binary CrossEntropy:
        since L = Σ 1/n -(ylog(σ(z)) + (1-y)log(1 - σ(z)))
        where:
            L = loss
            n = length of 1d array `z`
            y = labels
            log = natural logarithm
            σ = sigmoid function
            z = logits
        dL/dz = Σ 1/n -(y(σ(z))'/(σ(z)) + (1-y)(1 - σ(z))'/(1 - σ(z)))
              = Σ 1/n -(y(σ(z))(1 - σ(z))/(σ(z)) + (1-y)(-(σ(z))(1 - σ(z)))/(1 - σ(z))) since (σ(z))' = (σ(z))(1 - σ(z))
              = Σ 1/n -(y(1 - σ(z)) - (1-y)(σ(z)))
              = Σ 1/n -(y - y(σ(z)) - σ(z) + y(σ(z)))
              = Σ 1/n -(y  - σ(z))
              = Σ 1/n (σ(z) - y)
        the entire array is returned instead of the summation, as it will be dotted later.

        Categorical CrossEntropy:
        please see:
            https://deepnotes.io/softmax-crossentropy#derivative-of-cross-entropy-loss-with-softmax [1]

        the entire array is returned instead of the summation, as it will be dotted later.

        [1] P. Dahal, "Classification and Loss Evaluation - Softmax and Cross Entropy Loss", DeepNotes, 2020. [Online].
        Available: https://deepnotes.io/softmax-crossentropy#derivative-of-cross-entropy-loss-with-softmax.
        [Accessed: 16- Nov- 2020]

        """

        if self.isbinary and np.any(y > 1):
            raise Exception("Can not use 1 neuron for output layer of categorical classification task")
        return (sigmoid(z) - y)/len(z) if self.isbinary else (softmax(z) - one_hot(y, z.shape[1]))/len(z)


class MSE:
    """Calculates the rate of change of loss with respect to logits."""

    def __init__(self, isbinary=None):
        """
        Initialises Crossentropy.

        Parameters
        ----------
        isbinary : bool
            Initialised to None by default.

        """

        self.isbinary = isbinary
        pass

    def loss_gradient(self, z, y):
        """
        Calculates dL/dz where L represents Loss and z represents logits

        Parameters
        ----------
        z : ndarray
            1d Array containing logits of the current batch.
        y : ndArray
            1d Array containing labels of the current batch

        Returns
        -------
        ndarray
            1d Array of same shape as z and y containing dL/dz.

        Notes
        -----
        since L = 1/n Σ(y - z)^2
        where:
            L = loss
            n = length of 1d array `z`
            y = labels
            z = logits
        dL/dz = Σ 1/n 2*(y-z)*(-1)
              = Σ 2/n * (z-y)
        the entire array is returned instead of the summation, as it will be dotted later.

        """

        return 2/len(z) * (z-y)

