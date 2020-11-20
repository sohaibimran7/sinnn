import numpy as np


class Zeros:
    """Matrix of zeros."""
    def __init__(self):
        pass

    def initialise(self, rows, columns):
        """Returns a matrix of zeros.

        Parameters
        ----------
        rows : (int)
            The number of rows in the resulting matrix.
        columns : (int)
            The number of columns of the resulting matrix.

        Returns
        -------
        ndarray
            Array of zeros with shape: (`rows`, `columns`)

        """

        return np.zeros((rows, columns))


class SmallRandom:
    """ Matrix of random floats in the range 0 - lim"""

    def __init__(self):
        pass

    def initialise(self, rows, columns, lim=0.01):
        """Returns a matrix of random numbers in the range 0 - lim.

        Parameters
        ----------
        rows : (int)
            The number of rows in the resulting matrix.
        columns : (int)
            The number of columns of the resulting matrix.
        lim : (float)
            The upper limit of the random numbers.

        Returns
        -------
        ndarray
            Array of random numbers in range 0 - `lim` with shape: (`rows`, `columns`)

        """

        return np.random.randn(rows, columns) * lim
