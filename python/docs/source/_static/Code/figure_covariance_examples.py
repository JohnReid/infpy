#
# Copyright John Reid 2012
#

from numpy.random import seed
from infpy.gp import GaussianProcess, gp_1D_X_range, gp_plot_samples_from
from pylab import plot, savefig, title, close, figure, xlabel, ylabel
from infpy.gp import SquaredExponentialKernel as SE
from infpy.gp import Matern52Kernel as Matern52
from infpy.gp import Matern52Kernel as Matern32
from infpy.gp import RationalQuadraticKernel as RQ
from infpy.gp import NeuralNetworkKernel as NN
from infpy.gp import FixedPeriod1DKernel as Periodic
from infpy.gp import noise_kernel as noise

# seed RNG to make reproducible and close all existing plot windows
seed(2)
close('all')

#
# Part of X-space we will plot samples from
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
  gp_plot_samples_from(gp, support, num_samples=3)
  xlabel('x')
  ylabel('f(x)')
  title(fig_title)
  savefig('%s.png' % filename)
  savefig('%s.eps' % filename)
  
plot_for_kernel(
  kernel=Periodic(6.2),
  fig_title='Periodic',
  filename='covariance_function_periodic'
)

plot_for_kernel(
  kernel=RQ(1., dimensions=1),
  fig_title='Rational quadratic',
  filename='covariance_function_rq'
)

plot_for_kernel(
  kernel=SE([1]),
  fig_title='Squared exponential',
  filename='covariance_function_se'
)

plot_for_kernel(
  kernel=SE([3.]),
  fig_title='Squared exponential (long length scale)',
  filename='covariance_function_se_long_length'
)

plot_for_kernel(
  kernel=Matern52([1.]),
  fig_title='Matern52',
  filename='covariance_function_matern_52'
)

plot_for_kernel(
  kernel=Matern32([1.]),
  fig_title='Matern32',
  filename='covariance_function_matern_32'
)
