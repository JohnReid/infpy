#
# Copyright John Reid 2006, 2012
#


from real_kernel import *
import numpy
import math


class SquaredExponentialKernel(RealKernel):
    """
    A squared exponential kernel

    .. math::
        k(x_1, x_2) = k(r(x_1, x_2)) = \\textrm{exp}\\big(-\\frac{r^2}{2}\\big)

    where :math:`r(x_1, x_2) = |\\frac{x_1 - x_2}{l}|`
    """

    def __init__(self, params=None, priors=None, dimensions=None):
        (params, priors, dimensions) = kernel_init_args_helper(
            params,
            priors,
            dimensions
        )
        RealKernel.__init__(
            self,
            params,
            priors,
            dimensions
        )

    def __str__(self):
        return """SquaredExpKernel"""

    def __call__(self, x1, x2, identical=False):
        (x1, x2) = self._check_args(x1, x2)
        return math.exp(- distance_2(x1, x2, self.params) / 2)

    class Derivative(object):
        def __init__(self, k, i):
            self.k = k
            self.i = i

        def __call__(self, x1, x2, identical=False):
            (x1, x2) = self.k._check_args(x1, x2)
            return -(
                math.exp(- distance_2(x1, x2, self.k.params) / 2)
                / 2.0
                * distance_2_derivative(x1, x2, self.k.params, self.i)
            )


class ModulatedSquaredExponentialKernel(RealKernel):
    """
    Eq. 4.30 in `Gaussian Processes for Machine Learning`__ by Rasmussen and Williams. 

    __ http://www.amazon.co.uk/Gaussian-Processes-Learning-Adaptive-Computation/dp/026218253X/

    No trainable parameters
    """

    def __init__(self, sigma_g=1.0, sigma_u=1.0):
        self.g2 = sigma_g ** 2
        self.u2 = sigma_u ** 2
        self.e2 = 1.0 / (2.0 / self.g2 + 1.0 / self.u2)
        self.s2 = 2.0 * self.g2 + self.g2**2.0 / self.u2
        self.m2 = 2.0 * self.u2 + self.g2
        RealKernel.__init__(
            self,
            numpy.array([], numpy.float64),
            [],
            0
        )

    def __str__(self):
        return """ModulatedSquaredExpKernel"""

    def __call__(self, x1, x2, identical=False):
        (x1, x2) = self._check_args(x1, x2)
        d = len(x1)
        return (
            (self.e2 / self.u2) ** (d / 2.0)
            * math.exp(
                - numpy.dot(x1, x1) / (2.0 * self.m2)
                - numpy.dot(x1 - x2, x1 - x2) / (2.0 * self.s2)
                - numpy.dot(x2, x2) / (2.0 * self.m2)
            )
        )
