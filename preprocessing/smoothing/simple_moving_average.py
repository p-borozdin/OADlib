""" The module provides the implementation
of a simple moving average (SMA) smoothing algorithm
"""
import numpy as np
from OADlib.preprocessing.smoothing.base_smoother import BaseSmoother


class SimpleMovingAverage(BaseSmoother):
    """ Simple moving average (SMA) smoothing algorithms.\n
    The class inherits from BaseSmoother class
    """
    def __init__(
            self,
            window_size: int
            ):
        """ SMA algorithm initialization

        Args:
            window_size (int): size of the moving window.
                `window_size` must be greater that 0
        """
        assert window_size > 0, "window size expected to be > 0"
        super().__init__()

        self.window_size = window_size

    def execute(self, series: np.ndarray) -> np.ndarray:
        """ Executes left-hand SMA smoothing algorithm over a given series.\n
        Overriding the abstract method from the base class

        Args:
            series (np.ndarray): the series to smooth

        Returns:
            np.ndarray: smoothed series,
                the size is same as of the initiali series.

        First `window_size - 1` values are set to `numpy.nan`
        """
        padding = np.full(self.window_size - 1, np.nan)
        kernel = np.ones(self.window_size) / self.window_size
        smoothed = np.convolve(series, kernel, mode='valid')

        return np.concatenate((padding, smoothed))
