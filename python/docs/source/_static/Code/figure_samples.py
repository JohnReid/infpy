#
# Copyright John Reid 2012
#

from numpy.random import seed
from infpy.gp import GaussianProcess, gp_1D_X_range, gp_plot_samples_from
from pylab import plot, savefig, title, close, figure, xlabel, ylabel

# seed RNG to make reproducible and close all existing plot windows
seed(2)
close('all')

#
# Kernel
#
from infpy.gp import SquaredExponentialKernel as SE
kernel = SE([1])

#
# Part of X-space we will plot samples from
#
support = gp_1D_X_range(-10.0, 10.01, .125)

#
# Plot samples from prior
#
figure()
gp = GaussianProcess([], [], kernel)
gp_plot_samples_from(gp, support, num_samples=3)
xlabel('x')
ylabel('f(x)')
title('Samples from the prior')
savefig('samples_from_prior.png')
savefig('samples_from_prior.eps')

#
# Data
#
X = [[-5.], [-2.], [3.], [3.5]]
Y = [2.5, 2, -.5, 0.]

#
# Plot samples from posterior
#
figure()
plot([x[0] for x in X], Y, 'ks')
gp = GaussianProcess(X, Y, kernel)
gp_plot_samples_from(gp, support, num_samples=3)
xlabel('x')
ylabel('f(x)')
title('Samples from the posterior')
savefig('samples_from_posterior.png')
savefig('samples_from_posterior.eps')
