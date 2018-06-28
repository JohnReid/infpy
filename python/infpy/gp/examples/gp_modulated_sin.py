#
# Copyright John Reid 2009
#

import numpy
import math
import pylab
import infpy
import infpy.gp


def save_fig(prefix):
    pylab.savefig('%s.png' % prefix, format='PNG')
    pylab.savefig('%s.eps' % prefix, format='EPS')


# Generate some noisy data from a sin curve
# Our input domain
x_min, x_max = (10.0, 100.0)
X = infpy.gp.gp_1D_X_range(x_min, x_max, 1.0)
# noise-free output - shifted so mean=0
Y = [10.0 * math.sin(x[0]) / x[0] for x in X]
Y = infpy.gp.gp_zero_mean(Y)
# noise
e = 0.03 * numpy.random.normal(size=len(Y))
# noisy output
f = Y + e


# plot the noisy data
pylab.clf()
pylab.plot([x[0] for x in X], Y, 'b-', label='Y')
pylab.plot([x[0] for x in X], f, 'rs', label='f')
pylab.legend()
save_fig('gp-modulated-sin-data')
pylab.show()

# <demo> --- stop ---


def predict_values(K, learn=False):
    """Predict values for kernel K"""
    gp = infpy.gp.GaussianProcess(X, f, K)
    if learn:
        infpy.gp.gp_learn_hyperparameters(gp)
    pylab.clf()
    infpy.gp.gp_1D_predict(gp, 90, x_min - 1.0, x_max + 1.0, new_figure=False)


# create a kernel composed of a squared exponential kernel
# and a small noise term
K = infpy.gp.SquaredExponentialKernel() + infpy.gp.noise_kernel(0.1)
predict_values(K)
save_fig('gp-modulated-sin-se')


# <demo> --- stop ---


# Try a different kernel with a shorter characteristic length scale
K = infpy.gp.SquaredExponentialKernel([0.1]) + infpy.gp.noise_kernel(0.1)
predict_values(K)
save_fig('gp-modulated-sin-se-shorter')

# <demo> --- stop ---


# Try another kernel with a lot more noise
K = infpy.gp.SquaredExponentialKernel([4.0]) + infpy.gp.noise_kernel(1.0)
predict_values(K)
save_fig('gp-modulated-sin-more-noise')

# <demo> --- stop ---


# Try to learn kernel hyperparameters
K = infpy.gp.SquaredExponentialKernel([4.0]) + infpy.gp.noise_kernel(0.1)
predict_values(K, learn=True)
save_fig('gp-modulated-sin-learnt')

# <demo> --- stop ---
