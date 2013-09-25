#
# Copyright John Reid 2010
#

"""
Code to examine fixed points of a 2 component beta mixture under EM.
"""

import numpy as np, scipy as sc, pylab as pl, logging
from scipy.special import betaln, gammaln, digamma
from scipy.optimize import fmin_cg
from infpy.mixture import beta 


def plot_betaln():
    a, b = np.ogrid[.1:10:100j,.1:10:100j]
    z = betaln(a, b)
    pl.imshow(z, origin='lower', extent=[0,10,0,10])

def dbetaln_da(a, b):
    return digamma(a) + digamma(b) - digamma(a+b)

def plot_betaln_at(a):
    b = np.linspace(.1, 3, 1000)
    pl.plot(b, betaln(a, b), label='$\\log \\textrm{Beta}(%f, b)$' % a)
    pl.plot(b, dbetaln_da(a, b), ':', label='$\\frac{d\\log \\textrm{Beta}(%f, b)}{db}$' % a)
    #pl.legend()

def create_betaln_fig():
    pl.figure()
    plot_betaln_at(1.17)
    plot_betaln_at(10)
    pl.legend()
    
logging.basicConfig(level=logging.INFO)

np.seterr(over='warn')
np.random.seed(1)

N = 40
x = np.empty(N)
"Data."
x[:N/2] = np.random.beta(3, 1, N/2)
x[N/2:] = np.random.beta(1, 3, N/2)

T = np.empty((N, 2))
"Sufficient statistics of data."
T[:,0] = np.log(x)
T[:,1] = np.log(1-x)
weights = np.ones(N)


options = beta.get_default_options()
options.K = 2
options.point_estimates = True
options.max_iter = 25


family = beta.DirichletExpFamily(k=2)
exp_family = beta.DirichletExpFamily(k=2)
mixture = beta.ExpFamilyMixture(T, weights, options.K, exp_family, -np.ones(2), 1., options=options)
eta_0 = np.array([ 0.17656734,  0.16734223])
epsilon = 1e-1 * np.ones(2)
mixture.q_eta.eta[0] = eta_0 + epsilon
mixture.q_eta.eta[1] = eta_0 - epsilon

for _i in xrange(options.max_iter):
    mixture.update()
    bound = mixture.variational_bound()
    logging.info(
        'Iteration %3d: Variational bound = %e; eta distance = %e', 
        _i+1, bound, np.sqrt((mixture.q_eta.eta[0]-mixture.q_eta.eta[1])**2).sum()
    )

from cookbook.pylab_utils import set_rcParams_for_latex, get_fig_size_for_latex
set_rcParams_for_latex()
fig_size = get_fig_size_for_latex(1000)
pl.rcParams['figure.figsize'] = fig_size    
pl.figure()
mixture_x, density = mixture.plot(legend=True)
pl.savefig('mixture-fixed-point')
pl.close()

pl.rcParams['figure.figsize'] = fig_size    
pl.figure()
mixture_x, density = mixture.plot(legend=True, scale=False)
pl.savefig('mixture-fixed-point-unscaled')
pl.close()

#beta.plot_density_with_R(x, weights, 'mixture-fixed-point-R', mixture_x, density)
