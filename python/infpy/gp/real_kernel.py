#
# Copyright John Reid 2006, 2012
#

from kernel import *


class RealKernel(Kernel):
    """A kernel that is defined on :math:`\\mathbb{R}^n`"""

    def __init__(
            self,
            params,
            param_priors,
            dimensions
    ):
        """Initialises the kernel.

        If dimensions = 0 then the arguments to the kernel can take any
        dimension
        """
        Kernel.__init__(self, params, param_priors)
        self.dimensions = dimensions

    def _convert_arg(self, x):
        """Converts a kernel argument to the correct type.
        """
        if not hasattr(x, '__getitem__'):
            raise RuntimeError("""Data must be subscriptable: %s""" % str(x))
        if self.dimensions != 0 and self.dimensions != len(x):
            raise RuntimeError("""Data (%d) must be of dimension: %d"""
                               % (len(x), self.dimensions))
        return numpy.asarray(x, numpy.float64)

    def _check_args(self, x1, x2):
        """Checks that x1 and x2 have the right type and dimensions.

        If dimensions is specified also checks this agrees with arguments x1 and x2.

        Returns (x1, x2) converted to numpy.arrays with the correct type
        """
        x1 = self._convert_arg(x1)
        x2 = self._convert_arg(x2)
        assert type(x1) == type(x2) == numpy.ndarray
        return (x1, x2)


def distance_2(x1, x2, params):
    """Calculates distance squared between two points in :math:`\\mathbb{R}^n`

    .. math::
        r(x_1,x_2) = |\\frac{x_1 - x_2}{l}|^2

    where the parameters :math:`l` are the length scales.
    """
    assert type(x1) == type(x2) == numpy.ndarray
    if len(x1) != len(x2):
        raise RuntimeError(
            """Data points do not have the same dimension: %d and %d"""
            % (len(x1), len(x2)))
    if len(x1) != len(params):
        raise RuntimeError("""Data of dimension %d supplied. """
                           """This kernel is designed for data points of dimension: %d"""
                           % (len(x1), len(params)))
    diff = (x1 - x2) / params
    return numpy.dot(diff, diff)


def distance_2_derivative(x1, x2, params, i):
    """Calculates derivative of distance squared between two points w.r.t. param i
    """
    assert type(x1) == type(x2) == numpy.ndarray
    return -2.0 * (x1[i] - x2[i]) ** 2 / params[i] ** 3


def distance(x1, x2, params):
    """Calculates distance between two points in :math:`\\mathbb{R}^n`
    """
    import math
    assert type(x1) == type(x2) == numpy.ndarray
    return math.sqrt(distance_2(x1, x2, params))


def distance_derivative(x1, x2, params, i):
    """Calculates derivative of distance between two points w.r.t. param i
    """
    assert type(x1) == type(x2) == numpy.ndarray
    return -(
            ((x1[i] - x2[i]) ** 2)
        / (params[i] ** 3)
        / distance(x1, x2, params)
    )
