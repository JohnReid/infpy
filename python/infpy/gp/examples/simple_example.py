#
# Copyright John Reid 2008
#

import numpy
import pylab
import infpy.gp


def save_fig(prefix):
    "Save current figure in extended postscript and PNG formats."
    pylab.savefig('%s.png' % prefix, format='PNG')
    pylab.savefig('%s.eps' % prefix, format='EPS')


# Generate some noisy data from a modulated sin curve
x_min, x_max = 10.0, 100.0
X = infpy.gp.gp_1D_X_range(x_min, x_max)  # input domain
Y = 10.0 * numpy.sin(X[:, 0]) / X[:, 0]  # noise free output
Y = infpy.gp.gp_zero_mean(Y)  # shift so mean=0.
e = 0.03 * numpy.random.normal(size=len(Y))  # noise
f = Y + e  # noisy output

# plot the noisy data
pylab.figure()
pylab.plot(X[:, 0], Y, 'b-', label='Y')
pylab.plot(X[:, 0], f, 'rs', label='f')
pylab.legend()
save_fig('simple-example-data')
pylab.close()


def predict_values(K, file_tag, learn=False):
    "Create a GP with kernel K and predict values. Optionally learn K's hyperparameters if learn==True."
    gp = infpy.gp.GaussianProcess(X, f, K)
    if learn:
        infpy.gp.gp_learn_hyperparameters(gp)
    pylab.figure()
    infpy.gp.gp_1D_predict(gp, 90, x_min - 10., x_max + 10.)
    save_fig(file_tag)
    pylab.close()


# import short forms of GP kernel names
import infpy.gp.kernel_short_names as kernels

# create a kernel composed of a squared exponential kernel and a small noise term
K = kernels.SE() + kernels.Noise(.1)
predict_values(K, 'simple-example-se')

# Try a different kernel with a shorter characteristic length scale
K = kernels.SE([.1]) + kernels.Noise(.1)
predict_values(K, 'simple-example-se-shorter')

# Try another kernel with a lot more noise
K = kernels.SE([4.]) + kernels.Noise(1.)
predict_values(K, 'simple-example-more-noise')

# Try to learn kernel hyper-parameters
K = kernels.SE([4.0]) + kernels.Noise(.1)
predict_values(K, 'simple-example-learnt', learn=True)
