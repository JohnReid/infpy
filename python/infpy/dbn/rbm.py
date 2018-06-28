#
# Copyright John Reid 2007
#

import numpy.random
import numpy


def sigmoid(x):
    return 1.0 / (1.0 + numpy.exp(-x))


class RBM(object):
    """
    A restricted boltzman machine
    """

    def __init__(self, H, V):
        """
        Initialise the restricted boltzman machine with H hidden units and V
        visible units
        """
        self.H = H
        self.V = V
        self.init_random()

    def init_random(self, scale=1.0):
        """
        Randomly initialise the biases and weights
        """
        self.b = numpy.asmatrix(numpy.random.normal(
            loc=0.0, scale=scale, size=self.H))
        self.c = numpy.asmatrix(numpy.random.normal(
            loc=0.0, scale=scale, size=self.V))
        self.W = numpy.asmatrix(numpy.random.normal(
            loc=0.0, scale=scale, size=(self.H, self.V)))

    def Q_h(self, v):
        """
        Compute Q(h=1|v)
        """
        return sigmoid(- self.b - self.W * numpy.asmatrix(v).T)

    def P_v(self, h):
        """
        Compute P(v=1|h)
        """
        return sigmoid(- self.c - numpy.asmatrix(h) * self.W)

    def update(
            self,
            v
    ):
        """
        Update a restricted boltzman machine
        """
        Q_h = self.Q_h(v)
        h = Q_h > r.uniform(size=Q_h.shape)
        return h


if '__main__' == __name__:
    rbm = RBM(H=2, V=3)
    v = numpy.ones((5, 3))
    rbm.update(v)
