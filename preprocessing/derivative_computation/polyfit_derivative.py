""" The module provides the implementation
of a polynominal fitting derivative computation algorithm
"""
import numpy as np
from OADlib.preprocessing.derivative_computation.\
    base_derivative_computer import BaseDerivativeComputer


class PolyfitDerivative(BaseDerivativeComputer):
    """ Polynominal fitting derivative computation algorithm.
    Note that the "left" derivative is computed.\n
    The class inherits from BaseDerivativeComputer class
    """
    def __init__(
            self,
            poly_deg: int,
            n_points: int
            ):
        """ Polynominal fitting derivative computation algorithm initialization

        Args:
            poly_deg (int): polynominal degree to fit the points
            n_points (int): number of points used for fitting

        `poly_deg` and `n_points` must be related by ratio:
        `poly_deg <= n_points - 1`. Otherwise the fittng will be ambiguous
        """
        assert poly_deg <= n_points - 1, \
            f"For points = {n_points} the polynominal " \
            f"degree must be <= {n_points - 1}"

        super().__init__()

        self.poly_deg = poly_deg
        self.n_points = n_points

    def compute(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """ Computes the left-hand derivative `dy/dx` at each point using
        polynominal fitting\n
        Overriding the abstract method from the base class

        Args:
            x (np.ndarray): values on the abscissa axis (x-axis)
            y (np.ndarray): values on the ordinate axis (y-axis)

        Returns:
            np.ndarray: values of derivative `dy/dx`

        First `n_points - 1` values are set to `numpy.nan`,
        where `n_points` is the number of points used for fitting
        """
        assert x.size == y.size, \
            f"sizes of x and y must be the same, \
            but got x.size = {x.size}, and y.size = {y.size}"

        derivatives = np.full(x.size, np.nan)
        for i in range(self.n_points - 1, x.size):
            poly_coeffs = np.polyfit(
                x=x[i - (self.n_points - 1):i + 1],
                y=y[i - (self.n_points - 1):i + 1],
                deg=self.poly_deg
                )
            derivative = np.poly1d(poly_coeffs).deriv()
            derivatives[i] = derivative(x[i])

        return derivatives
