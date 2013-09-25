#
# Copyright John Reid 2010
#

"""
Code to plot the log likelihood for a beta mixture.
"""


import logging
from optparse import OptionParser
import pylab as pl, numpy as np
from cookbook.script_basics import log_options, setup_logging
import infpy.mixture.beta; reload(infpy.mixture.beta)
from infpy.mixture import beta



def sigmoid(x):
    return 1. / (1. + np.exp(-x))
    

setup_logging()
np.seterr(over='warn', invalid='raise')

parser = OptionParser()
beta.add_options(parser)
options, args = parser.parse_args()
log_options(parser, options)

logging.info('Seeding numpy.random')
np.random.seed(1)

exp_family = beta.DirichletExpFamily(k=2)

logging.info('Creating data')
block_size = 30
y = np.empty(3 * block_size)
y[:block_size] = np.random.normal(loc=-10, scale=2., size=block_size)
y[block_size:-block_size] = np.random.normal(loc=0, scale=4., size=block_size)
y[-block_size:] = np.random.normal(loc=10, scale=2., size=block_size)
x = sigmoid(y)
X = np.empty((len(x), 2))
X[:,0] = x
X[:,1] = 1.-X[:,0]
T = exp_family.T(X)
weights = np.random.rand(len(y))

logging.info('Creating new model.')
mixture = beta.ExpFamilyMixture(T, weights, options.K, exp_family, -np.ones(2), 1., options=options)

logging.info('Plotting LL.')
mixture.plot_ll(np.ones(len(y))/len(y), 10.)
pl.show()
