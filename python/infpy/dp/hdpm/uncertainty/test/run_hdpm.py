#
# Copyright John Reid 2010
#

"""
Code to run HDPM.
"""

import logging, numpy

import infpy.dp.hdpm.math
reload(infpy.dp.hdpm.math)
import infpy.dp.hdpm.uncertainty
reload(infpy.dp.hdpm.uncertainty)
import infpy.dp.hdpm.uncertainty as U
import infpy.dp.hdpm.uncertainty.summarise
reload(infpy.dp.hdpm.uncertainty.summarise)
from infpy.dp.hdpm.uncertainty.summarise import Statistics, InferenceHistory, Summariser
from infpy.convergence_test import LlConvergenceTest

logging.basicConfig(level=logging.DEBUG)

F, K, G, average_n_g = (12, 80, 100,  50)

numpy.random.seed(2)
logging.debug('Testing sampled data with F=%d; K=%d; G=%d, average n_g=%d', F, K, G, average_n_g)

options = U.get_default_options()
options.a_tau = numpy.ones(F)
options.a_omega = numpy.ones(F)

rho = U.sample_rho(G, average_n_g=average_n_g)
sample = U.sample(options, rho, K, F)

genes = U.genes_from_sites(sample.sites, rho)
data = U.Data(genes, F, options)
dist = U.VariationalDistribution(data, K)

Summariser(dist, 'output/sampled/summary').summarise_all()

history = InferenceHistory(dist)
LL = dist.log_likelihood()
LL_tolerance = 1e-8*data.N
logging.info('Tolerance in LL: %e', LL_tolerance)
max_iters = 50
convergence_test = LlConvergenceTest(eps=LL_tolerance, use_absolute_difference=True)
for i in xrange(max_iters):
    dist.update()
    history.update()
    LL = dist.log_likelihood()
    logging.info('Iteration: % 3d; LL = %e', i, LL)
    if convergence_test(LL):
        break
history.make_plots('output')

summariser = Summariser(dist, 'output/summary')
summariser.summarise_all()
