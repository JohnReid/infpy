#
# Copyright John Reid 2010
#


"""
Sample from infinite multinomials. Useful for Dirichlet processes.
"""

import bisect
import logging
import scipy.stats as S
import itertools
import numpy.random as R
import pylab as P
import numpy as N
import cookbook.pylab_utils as pylab_utils


# def rBeta(alpha, beta):
#    "Draw from a beta distribution."
#    x = S.beta.rvs(alpha, beta)
#    if N.isnan(x):
#        raise RuntimeError('Cannot draw from beta distribution with parameters: (%f, %f)' % (alpha, beta))
#    return x


def rBeta(alpha, beta):
    from rpy2.robjects import r
    x = r.rbeta(1, alpha, beta)
    if N.isnan(x):
        raise RuntimeError(
            'Cannot draw from beta distribution with parameters: (%f, %f)' % (alpha, beta))
    return x[0]


class LazyInfiniteSequence(list):
    """
    A lazy infinite sequence that extends itself as necessary.

    Sub-classes should implement self._extend() which returns next value to extend by.
    """

    def __init__(self, extender=None):
        "Initialise."
        if None == extender:
            extender = self._extend
        self.extender = extender

    def __getitem__(self, idx):
        "Get the k'th item, extending lazily if necessary."
        while len(self) <= idx:
            top = self.extender()
            logging.debug("Extending lazy infinite (%d) sequence with %s. Now length = %d.", id(
                self), str(top), len(self))
            self.append(top)
        return list.__getitem__(self, idx)

    def last(self):
        "@return: the last value."
        return list.__getitem__(self, -1)


class InfiniteMultinomial(LazyInfiniteSequence):
    """
    A lazy parameterisation of a multinomial over an infinite number of items.
    """

    def __init__(self):
        "Initialise."
        LazyInfiniteSequence.__init__(self)
        self.append(0.)

    def _check_invariants(self):
        "Make sure invariants aren't violated."
        assert len(self)
        assert 0. == self[0]
        for i in range(self.num_partitions()):
            assert self.partition_start(i) <= self.partition_end(i)
        assert self.last() <= 1.

    def partition(self, p):
        "Get the partition that contains p. Do this lazily from the infinite multinomial."
        # if p is in one of the partitions we already have, we just need to find the partition by binary search
        top = self.last()
        if p < top:
            partition = bisect.bisect(self, p) - 1
        else:
            # otherwise we need to keep extending number of partitions until we have one
            for k in itertools.count(self.num_partitions()):
                top = self.partition_end(k)
                if top > p:
                    break
            partition = self.num_partitions() - 1
        self._check_invariants()
        return partition

    def draw(self):
        "Sample randomly from the infinite multinomial."
        return self.partition(R.uniform())

    def num_partitions(self):
        "@return: The number of partitions we have so far."
        return len(self) - 1

    def partition_start(self, j):
        "@return: The start of the partition."
        return self[j]

    def partition_end(self, j):
        "@return: The end of the partition."
        return self[j + 1]

    def x(self, j):
        "@return: The x for the given partition. That is the weight of that partition."
        return self[j + 1] - self[j]


class StickBreaker(InfiniteMultinomial):
    """
    Models the stick breaking construction in a HDPM. Does its sampling in a lazy manner.

    A stick breaking construction models an infinite multinomial.
    """

    def __init__(self, gamma):
        "Initialise."
        InfiniteMultinomial.__init__(self)
        self.gamma = gamma

    def _extend(self):
        "Extend the stick breaking construction by one partition."
        pi_k = rBeta(1., self.gamma)
        top = self.last()
        top = top + (1. - top) * pi_k
        return top


class InfiniteDirichlet(InfiniteMultinomial):
    """
    Lazy sampling from an infinite Dirchlet distribution.
    """

    def __init__(self, strength, alpha):
        """
        Initialise with a strength parameter and an infinite prior. The infinite prior should be an infinite multinomial
        parameterisation itself.
        """
        InfiniteMultinomial.__init__(self)
        self.strength = strength
        self.alpha = alpha

    def _extend(self):
        "Extend the stick breaking construction by one partition."
        j = self.num_partitions()  # new partition is just past end of existing ones
        phi_j = rBeta(self.strength * self.alpha.x(j),
                      self.strength * (1. - self.alpha.partition_start(j + 1)))
        assert 0. <= phi_j
        assert phi_j <= 1.
        top = self.last()
        top = top + (1. - self.partition_end(j - 1)) * phi_j
        return top


def plot_multinomials(multis, kwargs_list=None):
    "Plot multinomials next to one another."
    K = max(list(map(len, multis)))
    M = len(multis)
    spacing = .3
    width = (1. - spacing) / M
    if None == kwargs_list:
        kwargs_list = itertools.repeat(dict())
    for n, (multi, kwargs) in enumerate(zip(multis, kwargs_list)):
        P.bar(N.arange(K) + (-width * M / 2 + n * width),
              [multi.x(k) for k in range(K)], width, **kwargs)


if '__main__' == __name__:

    logging.basicConfig(level=logging.DEBUG)

    R.seed(5)

    #
    # Test stick breaking
    #
    stick = StickBreaker(10.)
    for i in range(20):
        p = R.uniform()
        partition = stick.partition(p)
        logging.info('%f is in partition %d', p, partition)

    logging.info('Trying infinite dirichlet with stick breaking prior.')
    infinite_dirichlet = InfiniteDirichlet(1., stick)
    for i in range(20):
        p = R.uniform()
        partition = infinite_dirichlet.partition(p)
        logging.info('%f is in partition %d', p, partition)

    for partition in range(infinite_dirichlet.num_partitions()):
        logging.info('Partition %2d: stick: %.5f   infinite dirichlet: %.5f',
                     partition, stick.x(
                         partition), infinite_dirichlet.x(partition)
                     )

    strong_infinite_dirichlet = InfiniteDirichlet(10., stick)
    for i in range(20):
        p = R.uniform()
        partition = strong_infinite_dirichlet.partition(p)

    weak_infinite_dirichlet = InfiniteDirichlet(.01, stick)
    for i in range(20):
        p = R.uniform()
        partition = weak_infinite_dirichlet.partition(p)

    P.figure()
    plot_multinomials(
        (stick, strong_infinite_dirichlet, weak_infinite_dirichlet),
        ({'color': 'r'}, {'color': 'b'}, {'color': 'g'})
    )
    P.show()
