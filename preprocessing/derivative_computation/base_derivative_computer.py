""" The module provides the implementation of
a base abstract class for derivative computing algorithms
"""
import numpy as np


class BaseDerivativeComputer:
    """ Base class of a derivative computing algorithm
    """
    def compute(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """ Abstract method of calculating derivative y'(x) at each point

        Args:
            x (np.ndarray): values on the abscissa axis (x-axis)
            y (np.ndarray): values on the ordinate axis (y-axis)

        Raises:
            NotImplementedError:  call of a abstract method from base class

        Returns:
            np.ndarray: values of derivative `dy/dx`
        """
        raise NotImplementedError()
