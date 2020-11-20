import numpy as np
from sinnn.utils import sigmoid, softmax, one_hot


def loss(z, y, loss_function, isbinary=None):
    """
    Calculates the loss given the loss function used.

    Parameters
    ----------
    z : ndarray
        Array containing logits.
    y : ndArray
        Array containing labels.
    loss_function : str
        Name of loss function.
    isbinary : bool
        Whether the loss function should calculate Binary CrossEntropy or Categorical CrossEntropy

    Returns
    -------
    float64
        mean loss os of z from y

    Notes
    -----

    MSE:
        L = Σ 1/n (y - z)^2
        where:
            L = loss
            n = length of 1d array `z`
            y = labels
            z = logits

    CrossEntropy:

        Binary CrossEntropy:
        L = Σ 1/n -(ylog(σ(z)) + (1-y)log(1 - σ(z)))
        where:
            L = loss
            n = length of 1d array `z`
            y = labels
            log = natural logarithm
            σ = sigmoid function
            z = logits

        Categorical CrossEntropy:
        L = Σ 1/n Σ -(ylog(σ(z)))
        where:
            L = loss
            n = length of array `z`
            y = one hot encoded labels
            log = natural logarithm
            σ = softmax function
            z = logits

    """
    if loss_function == 'MSE':
        return np.mean(np.square(y - z))
    elif loss_function == 'CrossEntropy':
        return -np.mean(np.multiply(y, np.log(sigmoid(z))) + np.multiply(1 - y, np.log(1 - sigmoid(z)))) \
            if isbinary else \
            -np.mean(np.sum(np.multiply(one_hot(y, z.shape[1]), np.log(softmax(z))), axis=1))
    else:
        return None


def accuracy(z, y, loss_function, isbinary):
    """
    Calculates the accuracy if loss_function is CrossEntropy.

    For binary labels, `z` will be sigmoided to normalise their values and then rounded to compare to labels to
    calculate accuracy. For categorical labels, the index of the output neuron that gives the largest value is compared
    to labels to calculate accuracy. None is returned if loss_function does not support accuracy.

    Parameters
    ----------
    z : ndarray
        Array containing logits.
    y : ndArray
        Array containing labels.
    loss_function : str
        Name of loss function to verify if accuracy is supported.
    isbinary : bool
        Whether the accuracy should be calculated for Binary or Categorical labels.

    Returns
    -------
    float64
        accuracy of z with respect to y if `loss_function` supports accuracy else None.

    """
    return ((np.mean(np.round(sigmoid(z)) == y)
             if isbinary else
             np.mean(np.argmax(z, axis=1) == y))
            if loss_function == 'CrossEntropy' else
            None)
