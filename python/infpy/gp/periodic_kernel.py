#
# Copyright John Reid 2006, 2012
#


from .kernel import *
from .real_kernel import *
import numpy
import math


class FixedPeriod1DKernel(RealKernel):
    """See 4.31 in `Gaussian Processes for Machine Learning`__ by Rasmussen and Williams. 

    __ http://www.amazon.co.uk/Gaussian-Processes-Learning-Adaptive-Computation/dp/026218253X/

    .. math::

        k(x_1, x_2) = e^\\frac{-2 \sin^2( \\pi (x_1 - x_2) )}{p^2}

    Could be generalised to more than one dimension or for parameterisable periods
    """

    def __init__(
            self,
            fixed_period,
            length=1.0,
            prior=infpy.LogNormalDistribution()
    ):
        self.period = fixed_period
        RealKernel.__init__(
            self,
            numpy.asarray([length]),
            [prior],
            0
        )

    def d(self, x1, x2):
        return math.pi * (x1[0] - x2[0]) / self.period

    def __str__(self):
        return """PeriodicKernel( period = %f )""" % self.period

    def __call__(self, x1, x2, identical=False):
        (x1, x2) = self._check_args(x1, x2)
        return math.exp(
            - 2.0
            * math.sin(self.d(x1, x2)) ** 2
            / self.params[0] ** 2
        )

    class Derivative(object):
        def __init__(self, k, i):
            self.k = k
            assert 0 == i

        def __call__(self, x1, x2, identical=False):
            (x1, x2) = self.k._check_args(x1, x2)
            d = self.k.d(x1, x2)
            return (
                4.0 * math.sin(d) ** 2
                * math.exp(
                    - 2.0
                    * math.sin(d) ** 2
                    / self.k.params[0] ** 2
                )
                / self.k.params[0] ** 3
            )
