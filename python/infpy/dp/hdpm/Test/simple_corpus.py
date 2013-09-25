#
# Copyright John Reid 2010
#


"""
Define a simple corpus.

The corpus consists of 2 documents over a vocabulary of 2 words. The first document consists entirely of the
first word and the second the second word.
"""

import infpy.dp.hdpm as hdpm
import pylab as P, numpy, logging, itertools
from cookbook import pylab_utils
from infpy.convergence_test import LlConvergenceTest
from optparse import OptionParser
from basics import *

W = 2
"Size of vocabulary."

N = 100
"Number of words in each document."

documents = [[i] * N for i in xrange(W)]
"The documents."

K = 3 * W
"The maximum number of programs in the model."

def create_model():
    return hdpm.HDPM(documents, W, K)

