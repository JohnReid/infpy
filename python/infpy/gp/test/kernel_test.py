#
# Copyright John Reid 2006
#

from infpy.gp import *
import unittest
import numpy.random

num_tests = 100


def data_generator(dimensions):
    for i in range(num_tests):
        yield numpy.random.uniform(-2.0, 2.0, dimensions)


def param_generator(num_params):
    for i in range(num_tests):
        yield numpy.random.uniform(0.0, 2.0, dimensions)


if False:
    class KernelTest(unittest.TestCase):
        """Test case for all kernels"""

        def setUp(self):
            from numpy import linspace
            self.kernels = [
                (
                    'NeuralNetworkKernel',  # name
                    lambda p: NeuralNetworkKernel(
                        numpy.identity(2, numpy.float64)),  # to create
                    2,  # data dimensions
                    0,  # parameters
                ),
                (
                    'SquaredExponentialKernel',  # name
                    lambda p: SquaredExponentialKernel(params=p),  # to create
                    3,  # data dimensions
                    3,  # parameters
                ),
                (
                    'ModulatedSquaredExponentialKernel',  # name
                    lambda p: ModulatedSquaredExponentialKernel(),  # to create
                    5,  # data dimensions
                    0,  # parameters
                ),
                (
                    'RationalQuadraticKernel',  # name
                    lambda p: RationalQuadraticKernel(
                        alpha=1.0, params=p),  # to create
                    3,  # data dimensions
                    3,  # parameters
                ),
                (
                    'RationalQuadraticKernelAlphaParameterised',  # name
                    lambda p: RationalQuadraticKernelAlphaParameterised(
                        params=p),  # to create
                    3,  # data dimensions
                    4,  # parameters
                ),
                (
                    'PiecewisePoly0Kernel',  # name
                    lambda p: PiecewisePoly0Kernel(params=p),  # to create
                    3,  # data dimensions
                    3,  # parameters
                ),
                (
                    'PiecewisePoly1Kernel',  # name
                    lambda p: PiecewisePoly1Kernel(params=p),  # to create
                    3,  # data dimensions
                    3,  # parameters
                ),
                (
                    'Matern32Kernel',  # name
                    lambda p: Matern32Kernel(params=p),  # to create
                    3,  # data dimensions
                    3,  # parameters
                ),
                (
                    'Matern52Kernel',  # name
                    lambda p: Matern52Kernel(params=p),  # to create
                    3,  # data dimensions
                    3,  # parameters
                ),
                (
                    'noise_kernel',  # name
                    noise_kernel,  # to create
                    3,  # data dimensions
                    1,  # parameters
                ),
                (
                    'SumKernel',  # name
                    lambda p:  # to create
                    SumKernel(
                        SquaredExponentialKernel(params=p[:2]),
                        noise_kernel(p[2:])
                    ),
                    2,  # data dimensions
                    3,  # parameters
                ),
                (
                    'FixedPeriod1DKernel',  # name
                    lambda p: FixedPeriod1DKernel(1.0, p[0]),
                    1,  # data dimensions
                    1,  # parameters
                ),
            ]

        def tearDown(self):
            del self.kernels

        def testParameterFixer(self):
            for name, kernel_func, data_dims, num_params in self.kernels:
                # print name
                for x1, x2, params in zip(
                        data_generator(data_dims),
                        data_generator(data_dims),
                        param_generator(num_params)
                ):
                    assert (
                        kernel_func(params)(x1, x2)
                        ==
                        KernelParameterFixer(kernel_func(params))(x1, x2)
                    )

        def testGradient(self):
            for name, kernel_func, data_dims, num_params in self.kernels:
                # print name
                for x1, x2, params in zip(
                        data_generator(data_dims),
                        data_generator(data_dims),
                        param_generator(num_params)
                ):
                    check_gradients(
                        # create kernel and call on x1,x2
                        lambda t: kernel_func(t)(x1, x2),
                        lambda t: [  # create kernel and call derivative for each param on x1,x2
                            kernel_func(t).derivative_wrt_param(i)(x1, x2)
                            for i in range(len(t))
                        ],
                        params
                    )


class RealKernelTest(unittest.TestCase):
    """Test case for all kernels on real vector spaces"""

    def setUp(self):
        self.kernel_constructors = [
            lambda dims: RationalQuadraticKernel(alpha=1.0, dimensions=dims),
            lambda dims: RationalQuadraticKernelAlphaParameterised(
                dimensions=dims),
            lambda dims: SquaredExponentialKernel(dimensions=dims),
            lambda dims: ModulatedSquaredExponentialKernel(),
            lambda dims: PiecewisePoly0Kernel(dimensions=dims),
            lambda dims: PiecewisePoly1Kernel(dimensions=dims),
            lambda dims: Matern32Kernel(dimensions=dims),
            lambda dims: Matern52Kernel(dimensions=dims),
            lambda dims: NeuralNetworkKernel(sigma=numpy.identity(dims)),
            lambda dims: FixedPeriod1DKernel(fixed_period=1.0),
            lambda dims: noise_kernel(),
            lambda dims: SumKernel(
                Matern32Kernel(dimensions=dims),
                ConstantKernel(1.0)),
            lambda dims: ProductKernel(
                Matern32Kernel(dimensions=dims),
                ConstantKernel(3.0)),
        ]

    def testStr(self):
        """Test string representation of kernel"""
        [
            str(kernel_constructor(2))
            for kernel_constructor
            in self.kernel_constructors
        ]

    def testGradients(self):
        """Test the gradients of the kernel"""
        # for each kernel
        for kernel_constructor in self.kernel_constructors:
            # for each number of dimensions we want to test
            for dims in [1, 2, 3, 4]:
                # construct the kernel
                k = kernel_constructor(dims)

                # positions of our test data points
                mean = numpy.zeros(dims, numpy.float64)
                cov = numpy.identity(dims, numpy.float64)

                def test_point():
                    return numpy.random.multivariate_normal(mean, cov)

                # parameters at which to test
                params = [
                    None == prior and 1.0 or prior.sample_from()
                    for prior in k.param_priors
                ]

                # two test points
                x1, x2 = (test_point(), test_point())

                # function evaluated at this parameter and test points
                def f(p):
                    k.set_params(p)
                    return numpy.asarray([k(x1, x2)])

                # derivative
                def fprime(p):
                    k.set_params(p)
                    return numpy.asarray([
                        k.derivative_wrt_param(i)(x1, x2)
                        for i in xrange(len(p))
                    ])

                try:
                    # check the calculated gradient matches the approximation by expansion
                    infpy.check_gradients(f, fprime, params)

                except:
                    print 'Problem with %s' % str(k)
                    print 'Parameters = %s' % str(params)
                    print 'x1 = %s' % str(x1)
                    print 'x2 = %s' % str(x2)
                    raise


if __name__ == "__main__":
    unittest.main()
