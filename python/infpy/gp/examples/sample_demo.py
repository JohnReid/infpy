#
# Copyright John Reid 2006
#

import infpy.gp
import numpy.random
import pylab

numpy.random.seed(1)


SE = infpy.gp.SquaredExponentialKernel
Constant = infpy.gp.ConstantKernel
Noise = infpy.gp.noise_kernel


def sample_and_plot(k, *args, **kwds):
    support = infpy.gp.gp_1D_X_range(-10.0, 10.0, .1)
    gp = infpy.gp.GaussianProcess([[0.0]], [numpy.random.normal()], k)
    sample = infpy.gp.gp_sample_from(gp, support)
    pylab.plot([x[0] for x in support], sample, *args, **kwds)


def show_examples(examples, title):
    pylab.clf()
    for k, label in examples:
        sample_and_plot(k, label=label)

    pylab.title(title)
    pylab.legend()
    pylab.show()


def multiple_draws_examples():
    for i in xrange(4):
        yield SE([1]), 'function %d' % (i + 1)


show_examples(multiple_draws_examples(), 'Draws from GP prior')
# <demo> --- stop ---


def se_examples():
    for ls in [4.0, 1.0, 0.1]:
        yield SE([ls]), 'length scale = %f' % ls


show_examples(se_examples(), 'Squared exponential covariances')
# <demo> --- stop ---


# RQ = infpy.gp.RationalQuadraticKernel
# def rq_examples():
#       for alpha in [ 0.01, .1, 1.0 ]:
#               yield RQ( alpha, [ 1.0 ] ), 'alpha = %f' % alpha
#
# show_examples( rq_examples(), 'Rational quadratic covariances' )
# # <demo> --- stop ---
#
#
#
# Matern = infpy.gp.Matern32Kernel
# def matern_examples():
#       for ls in [ 4.0, 1.0, 0.1 ]:
#               yield Matern( [ ls ] ), 'length scale = %f' % ls
#
# show_examples( matern_examples(), 'Matern covariances' )
# # <demo> --- stop ---
#
#

Period = infpy.gp.FixedPeriod1DKernel


def periodic_examples():
    for period in [8.0, 4.0, 1.0]:
        yield Period([period]), 'period = %f' % period


show_examples(periodic_examples(), 'Periodic covariances')
# <demo> --- stop ---


def combination_examples():
    yield Period([4.0]) + Noise(0.1), 'period(4) + noise(0.1)'
    yield Period([4.0]) * SE([5]), 'period(4) * se(lengthscale=5)'


show_examples(combination_examples(), 'Covariance function combinations')
