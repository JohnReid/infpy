#
# Copyright John Reid 2010
#


"""
Code to test HDPM initialisation methods.
"""


import infpy.dp.gen_artificial_data as G
import logging
import numpy.random
import basics
import numpy as N
import infpy.dp.hdpm as hdpm

logging.basicConfig(level=logging.INFO)

parser = basics.create_option_parser()
parser.add_option(
    "-n",
    dest="num_models_to_test",
    default=10,
    type='int',
    help="Number of models to test."
)
options, args = basics.parse_options(parser)
numpy.random.seed(options.seed)

G.seed(options.seed)
W = 5
D = 20
a_alpha, b_alpha = 20., 10.
a_beta, b_beta  = 20., 10.
a_gamma, b_gamma = 20., 10.
a_tau = N.ones(W)
document_sizes = numpy.random.poisson(100., D)
sampler = G.HDPMSampler(
    document_sizes,
    a_alpha, b_alpha,
    a_beta, b_beta,
    a_gamma, b_gamma,
    a_tau
)
logging.info(
    'Sampled from HDPM: W=%d; D=%d; N=%d; K=%d',
    sampler.W(), sampler.D(), sampler.N(), sampler.K()
)


logging.info('Running sampled models.')
best_sample_LL = None
for i in xrange(options.num_models_to_test):
    sampled_model = hdpm.HDPM(
        sampler.get_documents(),
        W,
        sampler.K() * 3,
        a_alpha, b_alpha,
        a_beta, b_beta,
        a_gamma, b_gamma,
        a_tau,
        init_method=hdpm.HDPM.INIT_METHOD_SAMPLE
    )
    LL, model = basics.infer_model(sampled_model, options)
    if None == best_sample_LL or LL > best_sample_LL:
        best_sample_LL = LL

logging.info('Running simply initialised models.')
best_simple_LL = None
for i in xrange(options.num_models_to_test):
    sampled_model = hdpm.HDPM(
        sampler.get_documents(),
        W,
        sampler.K() * 3,
        a_alpha, b_alpha,
        a_beta, b_beta,
        a_gamma, b_gamma,
        a_tau,
        init_method=hdpm.HDPM.INIT_METHOD_SIMPLE
    )
    LL, model = basics.infer_model(sampled_model, options)
    if None == best_simple_LL or LL > best_simple_LL:
        best_simple_LL = LL

logging.info(
    'Sampled initialisation gave best LL: %.3f, simple initialisation gave best LL: %.3f',
    best_sample_LL, best_simple_LL
)
