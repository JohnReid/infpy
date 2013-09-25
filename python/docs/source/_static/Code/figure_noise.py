#
# Copyright John Reid 2012
#

from numpy.random import seed
from infpy.gp import GaussianProcess, gp_1D_X_range, gp_plot_prediction
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

def plot_for_kernel(kernel, fig_title, filename):
  figure()
  plot([x[0] for x in X], Y, 'ks')
  gp = GaussianProcess(X, Y, kernel)
  mean, sigma, LL = gp.predict(support)
  gp_plot_prediction(support, mean, sigma)
  xlabel('x')
  ylabel('f(x)')
  title(fig_title)
  savefig('%s.png' % filename)
  savefig('%s.eps' % filename)
  
plot_for_kernel(
  kernel=SE([1.]) + noise(.1),
  fig_title='k = SE + noise(.1)',
  filename='noise_mid'
)

plot_for_kernel(
  kernel=SE([1.]) + noise(1.),
  fig_title='k = SE + noise(1)',
  filename='noise_high'
)

plot_for_kernel(
  kernel=SE([1.]) + noise(.0001),
  fig_title='k = SE + noise(.0001)',
  filename='noise_low'
)

