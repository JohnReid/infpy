#
# Copyright John Reid 2009
#

import numpy
import pylab
import infpy.gp

# Generate some noisy data from a sin curve
# Our input domain
x_min, x_max = (0.0, 5.0)
X = infpy.gp.gp_1D_X_range(x_min, x_max, 0.3)
# noise-free output - shifted so mean=0
Y = numpy.sin([x[0] for x in X])
Y = infpy.gp.gp_zero_mean(Y)
# noise
e = 0.1 * numpy.random.normal(size=len(Y))
# noisy output
f = Y + e


# plot the noisy data
print X
pylab.plot([x[0] for x in X], Y, 'g--', label='Y')
pylab.plot([x[0] for x in X], f, 'ks', label='f')
pylab.legend()
pylab.show()

# <demo> --- stop ---


# create a kernel composed of a squared exponential kernel
# and a small noise term
K = infpy.gp.SquaredExponentialKernel() + infpy.gp.noise_kernel(0.1)

# Create a gaussian process
gp = infpy.gp.GaussianProcess(X, f, K)

# display predictions discretised over a number of points
infpy.gp.gp_1D_predict(gp, 90, x_min - 1.0, x_max + 1.0)

# <demo> --- stop ---


# Try a different kernel with a shorter characteristic length scale
K = infpy.gp.SquaredExponentialKernel([0.1]) + infpy.gp.noise_kernel(0.1)
gp = infpy.gp.GaussianProcess(X, f, K)
infpy.gp.gp_1D_predict(gp, 90, x_min - 1.0, x_max + 1.0)

# <demo> --- stop ---


# Try another kernel with a lot more noise
K = infpy.gp.SquaredExponentialKernel([4.0]) + infpy.gp.noise_kernel(1.0)
gp = infpy.gp.GaussianProcess(X, f, K)
infpy.gp.gp_1D_predict(gp, 90, x_min - 1.0, x_max + 1.0)

# <demo> --- stop ---


# Try to learn kernel hyperparameters
K = infpy.gp.SquaredExponentialKernel([4.0]) + infpy.gp.noise_kernel(0.1)
gp = infpy.gp.GaussianProcess(X, f, K)
infpy.gp.gp_learn_hyperparameters(gp)
infpy.gp.gp_1D_predict(gp, 90, x_min - 1.0, x_max + 1.0)

# <demo> --- stop ---
