""" The module provides the implementation
of a base abstract class for smoothing algorithms
"""
import numpy as np


class BaseSmoother:
    """ Base class of a smoothing algorithm
    """
    def execute(self, series: np.ndarray) -> np.ndarray:
        """ Abstract method of executing a smoothing algorithm
        over a given series

        Args:
            series (np.ndarray): the series to smooth

        Raises:
            NotImplementedError: call of a abstract method from base class

        Returns:
            np.ndarray: smoothed series
        """

        raise NotImplementedError()
