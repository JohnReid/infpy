#
# Copyright John Reid 2006
#


from .real_kernel import *
import numpy
import math


class RationalQuadraticKernel(RealKernel):
    """
    Following 4.19 in `Gaussian Processes for Machine Learning`__ by Rasmussen and Williams. 

    __ http://www.amazon.co.uk/Gaussian-Processes-Learning-Adaptive-Computation/dp/026218253X/

    .. math::

        k = (1 + \\frac{r^2}{2\\alpha})^{-\\alpha}

    Alpha is not a trainable parameter.
    The parameters are for the length scale in the r term.
    """

    def __init__(
            self,
            alpha,
            params=None,
            priors=None,
            dimensions=None
    ):
        self.alpha = alpha
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
        return """RQKernelFixedAlpha( alpha = %f )""" % self.alpha

    def __call__(self, x1, x2, identical=False):
        (x1, x2) = self._check_args(x1, x2)
        r2 = distance_2(x1, x2, self.params)
        return (1.0 + r2 / (2.0 * self.alpha)) ** (-self.alpha)

    class Derivative(object):
        def __init__(self, k, i):
            self.k = k
            self.i = i

        def __call__(self, x1, x2, identical=False):
            (x1, x2) = self.k._check_args(x1, x2)
            r2 = distance_2(x1, x2, self.k.params)
            term = 1.0 + r2 / (2.0 * self.k.alpha)
            return -(
                term ** (-self.k.alpha - 1.0)
                / 2.0
            ) * distance_2_derivative(x1, x2, self.k.params, self.i)


class RationalQuadraticKernelAlphaParameterised(RealKernel):
    """Following 4.19 in `Gaussian Processes for Machine Learning`__ by Rasmussen and Williams. 

    __ http://www.amazon.co.uk/Gaussian-Processes-Learning-Adaptive-Computation/dp/026218253X/

    .. math::

        k = (1 + \\frac{r^2}{2\\alpha})^{-\\alpha}

    The first parameter is alpha.
    The rest of the parameters are for the length scale in the r term.
    """

    def __init__(
            self,
            params=None,
            priors=None,
            dimensions=None
    ):
        (params, priors, dimensions) = kernel_init_args_helper(
            params,
            priors,
            dimensions,
            num_extra_params=1
        )
        RealKernel.__init__(
            self,
            params,
            priors,
            dimensions
        )

    def alpha(self): return self.params[0]

    def __call__(self, x1, x2, identical=False):
        (x1, x2) = self._check_args(x1, x2)
        r2 = distance_2(x1, x2, self.params[1:])
        return (1.0 + r2 / (2.0 * self.alpha())) ** (-self.alpha())

    def __str__(self):
        return """RQKernelAlphaParameterised( alpha = %f )""" % self.alpha()

    class Derivative(object):
        def __init__(self, k, i):
            self.k = k
            self.i = i

        def __call__(self, x1, x2, identical=False):
            (x1, x2) = self.k._check_args(x1, x2)
            r2 = distance_2(x1, x2, self.k.params[1:])
            term = 1.0 + r2 / (2.0 * self.k.alpha())
            if self.i == 0:  # alpha
                return (
                    -2.0 * self.k.alpha() * math.log(term) * term ** (-self.k.alpha())
                    + r2 * term ** (-self.k.alpha() - 1)
                ) / (
                    2 * self.k.alpha()
                )
            else:  # lengthscale
                return -(
                    term ** (-self.k.alpha() - 1.0)
                    / 2.0
                ) * distance_2_derivative(x1, x2, self.k.params[1:], self.i - 1)
