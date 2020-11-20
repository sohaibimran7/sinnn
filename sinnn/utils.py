import numpy as np
import pickle


def sigmoid(x):
    """
    Applies the sigmoid transformation on a given 1d vector

    Parameters
    ----------
    x : ndarray
        1d Array to be sigmoided

    Returns
    -------
    ndarray
        1d Array of same shape as x containing sigmoided values of x

    Notes
    -----
    σ(x) = 1/(1 + e^-x) = (e^x)/(1 + e^x)
    where:
        σ = sigmoid function
        x = input
        e = Euler's number

    """
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


def softmax(x):
    """
    Applies the softmax transformation on a given 2d vector

    In order to make softmax more stable, x.max is subtracted from each value of x before performing softmax.

    Parameters
    ----------
    x : ndarray
        2d array to be softmaxed

    Returns
    -------
    ndarray
        2d Array of same shape as x containing softmaxed values of x

    Notes
    -----
    σ(x) = (e^x_i)/ Σ (e^x)
    where:
        σ = softmax function
        x_i = ith value of x
        x = input
        e = Euler's number

    """

    x = x - x.max(axis=1, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=1, keepdims=True)


def one_hot(labels, categories):
    """
    One hot encodes the labels.

    Parameters
    ----------
    labels : ndarray
        1d array of ints containing labels.

    categories :  int
        the number of categories to encode the labels onto

    Returns
    -------
    ndarray
        2d Array of shape (len(x), categories) containing one hot encoded labels

    """
    y = np.zeros((labels.size, categories))
    y[np.arange(labels.size), labels] = 1
    return y


def batches(x, y, batch_size):
    """
    Yields successive batch_sized chunks of inputs and corresponding outputs from list.

    Parameters
    ----------
    x : list
        list of inputs
    y : list
        list of outputs
    batch_size :
        size of the batches to be yielded.

    Returns
    -------
    list
        batch of inputs of length (batch_size)
    list
        batch of outputs of length (batch_size)

    """

    for i in range(0, len(x), batch_size):
        yield x[i:i + batch_size], y[i:i + batch_size]


def save_model(model, filename='model'):
    """
    Saves an object to file

    Parameters
    ----------
    model : object
        any object to be stored on file
    filename : str
        name of the file the object needs to be stored on. Default: model

    """
    with open(filename, 'wb') as f:
        pickle.dump(model, f, protocol=-1)


def load_model(filename='model'):
    """
    Loads an object from file.

    Parameters
    ----------
    filename: str
        name of the file the object needs to be loaded from. Default: model

    Returns
    -------
        object that was stored on the file
    """
    with open(filename, 'rb') as f:
        return pickle.load(f)
