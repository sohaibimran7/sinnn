import numpy as np


class Initialiser:
    """ Initialises weights """

    def __init__(self):
        pass

    def initialise(self, rows, columns):
        pass


class Zeros(Initialiser):
    """ Returns a matrix of of zeros """

    def __init__(self):
        pass

    def initialise(self, rows, columns):
        return np.zeros((rows, columns))


class SmallRandom(Initialiser):
    """ Returns a matrix of random numbers in the range 0 - lim"""

    def __init__(self):
        pass

    def initialise(self, rows, columns, lim=0.01):
        return np.random.randn(rows, columns) * lim
