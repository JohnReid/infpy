#
# Copyright John Reid 2006
#

import infpy.gp
import numpy.random
import pylab

numpy.random.seed(1)


X = infpy.gp.gp_1D_X_range(-5, 5, .3)
F = numpy.array([x[0] ** 2 for x in X])
Y = F + 2.0 * numpy.random.normal(size=len(F))
Y_gp, revert = infpy.zero_mean_unity_variance(Y)


def plot_Y_gp():
    pylab.plot([x[0] for x in X], Y_gp, 'rs')


pylab.clf()
plot_Y_gp()

SE = infpy.gp.SquaredExponentialKernel
Constant = infpy.gp.ConstantKernel
Noise = infpy.gp.noise_kernel
Fix = infpy.gp.KernelParameterFixer

k_noise_free = Constant(1) * SE([1]) + Fix(Noise(0.01))
k_noisy = Constant(1) * SE([1]) + Noise(.2)


def predict(k):
    gp = infpy.gp.GaussianProcess(X, Y_gp, k)
    infpy.gp.gp_learn_hyperparameters(gp)
    infpy.gp.gp_1D_predict(
        gp,
        num_steps=100,
        x_min=X[0][0],
        x_max=X[-1][0],
        show_y=False,
        show_variance=False,
        new_figure=False
    )


predict(k_noise_free)
predict(k_noisy)
pylab.title('')
pylab.savefig('fit.png')
