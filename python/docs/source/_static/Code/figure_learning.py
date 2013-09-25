#
# Copyright John Reid 2012
#

from numpy.random import seed
from infpy.gp import GaussianProcess, gp_1D_X_range
from infpy.gp import gp_plot_prediction, gp_learn_hyperparameters
from pylab import plot, savefig, title, close, figure, xlabel, ylabel
from infpy.gp import SquaredExponentialKernel as SE
from infpy.gp import noise_kernel as noise

# close all existing plot windows
close('all')

#
# Part of X-space we are interested in
#
support = gp_1D_X_range(-10.0, 10.01, .125)

#
# Data
#
X = [[-5.], [-2.], [3.], [3.5]]
Y = [2.5, 2, -.5, 0.]

def plot_gp(gp, fig_title, filename):
  figure()
  plot([x[0] for x in X], Y, 'ks')
  mean, sigma, LL = gp.predict(support)
  gp_plot_prediction(support, mean, sigma)
  xlabel('x')
  ylabel('f(x)')
  title(fig_title)
  savefig('%s.png' % filename)
  savefig('%s.eps' % filename)
  
#
# Create a kernel with reasonable parameters and plot the GP predictions
#
kernel = SE([1.]) + noise(1.)
gp = GaussianProcess(X, Y, kernel)
plot_gp(
  gp=gp,
  fig_title='Initial parameters: kernel = SE([1]) + noise(1)',
  filename='learning_first_guess'
)

#
# Learn the covariance function's parameters and replot
#
gp_learn_hyperparameters(gp)
plot_gp(
  gp=gp,
  fig_title='Learnt parameters: kernel = SE([%.2f]) + noise(%.2f)' % (
    kernel.k1.params[0],
    kernel.k2.params.o2[0]
  ),
  filename='learning_learnt'
)
