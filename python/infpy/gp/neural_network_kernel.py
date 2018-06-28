#
# Copyright John Reid 2006, 2012
#


from kernel import *
from real_kernel import *
import numpy
import math


class NeuralNetworkKernel(RealKernel):
    """Following 4.29 in `Gaussian Processes for Machine Learning`__ by Rasmussen and Williams. 

    __ http://www.amazon.co.uk/Gaussian-Processes-Learning-Adaptive-Computation/dp/026218253X/

    No trainable paramaters
    """

    def __init__(
            self,
            sigma
    ):
        self.sigma = numpy.matrix(sigma, numpy.float64)
        if self.sigma.shape[0] != self.sigma.shape[1]:
            raise RuntimeError("""Sigma must be a square matrix of size equal to """
                               """dimension of the data: %d, %d""" % self.sigma.shape)
        RealKernel.__init__(
            self,
            numpy.asarray([]),
            [],
            self.sigma.shape[0]
        )

    def _calc_term(self, x1, x2):
        return 2.0 * x1 * self.sigma * x2.T

    def __str__(self):
        return """NeuralNetworkKernel"""

    def __call__(self, x1, x2, identical=False):
        (x1, x2) = self._check_args(x1, x2)
        x1 = numpy.matrix(x1)
        x2 = numpy.matrix(x2)
        return (
            2.0 / math.pi
            * math.asin(
                self._calc_term(x1, x2)
                / math.sqrt(
                    self._calc_term(x1, x1)
                    * self._calc_term(x2, x2)
                )
            )
        )
