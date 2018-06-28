#
# Copyright John Reid 2010
#


"""
Code to test HDPM.
"""


import gen_artificial_data as G
import pylab as P
import itertools
import numpy as N
import logging
from cookbook import pylab_utils

logging.basicConfig(level=logging.INFO)

G.seed(3)
W = 3
sampler = G.HDPMSampler(
    N.arange(5) + 10,
    20., 10.,
    20., 10.,
    20., 10.,
    N.ones(W)
)
logging.info(
    'Sampled from HDPM: W=%d; D=%d; N=%d; K=%d',
    sampler.W(), sampler.D(), sampler.N(), sampler.K()
)


P.figure()
G.plot_multinomials(
    list(itertools.chain((sampler.pi,), [t for t, Z, X in sampler.documents])),
    map(lambda c: {'color': c},
                   itertools.cycle(pylab_utils.simple_colours))
)
P.title('Pi and thetas')
P.show()
