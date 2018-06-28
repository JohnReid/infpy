#
# Copyright John Reid 2006
#

"""Model periodic UK gas consumption data from R"""

import rpy
import math
import numpy
import pylab
import infpy.gp
import sys
import scipy.stats


def save_fig(prefix):
    pylab.savefig('%s.png' % prefix, format='PNG')
    pylab.savefig('%s.eps' % prefix, format='EPS')


# import and set data's mean to 0
input_data = rpy.r.UKgas[20:60]  # it is a little slow with all data
predict_max = len(input_data) / 4.0 + 5.0
# print len( input_data )
gp_Y, revert = infpy.zero_mean_unity_variance(input_data)
# make one year one unit on X-axis
gp_X = [[float(i) / 4.0] for i in range(len(input_data))]
# print gp_Y
pylab.clf()
pylab.plot(gp_Y, 'rs-')
pylab.title('Uk gas consumption data (source: R)')
save_fig('gp-uk-gas-data')
pylab.show()


#
# Create kernels to model different properties of the data
#
LN = infpy.LogNormalDistribution
Gamma = infpy.GammaDistribution
Constant = infpy.gp.ConstantKernel
Noise = infpy.gp.noise_kernel
SE = infpy.gp.SquaredExponentialKernel
RQ = infpy.gp.RationalQuadraticKernelAlphaParameterised
Periodic = infpy.gp.FixedPeriod1DKernel
Fix = infpy.gp.KernelParameterFixer


def show_kernel_predictions(k):
    # Create a gaussian process
    gp = infpy.gp.GaussianProcess(gp_X, gp_Y, k)
    pylab.clf()
    infpy.gp.gp_1D_predict(gp, x_max=predict_max, new_figure=False)
    print 'Parameters: %s\nLL: %f' % (str(gp.k.params), gp.LL)


def learn_kernel_parameters(k):
    # Create a gaussian process
    gp = infpy.gp.GaussianProcess(gp_X, gp_Y, k)
    infpy.gp.gp_learn_hyperparameters(gp)
    print 'Parameters: %s\nLL: %f' % (str(gp.k.params), gp.LL)
# <demo> --- stop ---


# first use a general trend term with some noise
k = (
    Constant(1.2) * Fix(SE([12]))  # gradual trend - n.b. long length-scale
    # noise
    + Noise(0.8)
)
# learn_kernel_parameters( k )
show_kernel_predictions(k)
save_fig('gp-uk-gas-general')
# <demo> --- stop ---

# Try and learn a periodic term
k = (
    Constant(1) * SE([3]) * Periodic(1, 1.5)
    + Noise(0.2)
)
# learn_kernel_parameters( k )
show_kernel_predictions(k)
save_fig('gp-uk-gas-periodic')
# <demo> --- stop ---

# Enforce a reasonable length scale
k = (
    Constant(3) * Fix(SE([12])) * Periodic(1, 1.8)
    + Noise(0.3)
)
#learn_kernel_parameters( k )
show_kernel_predictions(k)
save_fig('gp-uk-gas-periodic-reasonable-length')
# <demo> --- stop ---
