#
# Copyright John Reid 2007
#

"""
Code for exponential family harmoniums

Follows Welling, Rosen-Zvi, Hinton 2005
"""

from numpy import asmatrix, array, empty, matrix, zeros_like, zeros


class Harmonium(object):
    """
    An exponential family harmonium with hidden and observed variables

    Follows Welling, Rosen-Zvi, Hinton 2005
    """

    def __init__(self, observed, hidden):
        """
        Initialise with the observed and hidden variables
        """
        from numpy.random import gamma
        self.observed = array(observed)
        self.hidden = array(hidden)
        self.W = empty((len(observed), len(hidden)), dtype=numpy.object_)

        # randomise W
        for i, o in enumerate(observed):
            for j, h in enumerate(hidden):
                self.W[i, j] = asmatrix(-gamma(1.0,
                                               size=(o.family.size(), h.family.size())))

    def theta_hat(self, i, hidden_values):
        """
        The natural parameters of an observed variable taking quadratic term for
        interactions into account
        """
        assert len(hidden_values) == len(self.hidden)
        theta = matrix(self.observed[i].phi)
        for j, h in enumerate(hidden_values):
            assert self.hidden[j].family._check_u(h)
            update = self.W[i, j] * asmatrix(h).T
            theta += update.T
        return theta.A.reshape((self.observed[i].family.size()))

    def lambda_hat(self, j, observed_values):
        """
        The natural parameters of a hidden variable taking quadratic term for
        interactions into account
        """
        assert len(observed_values) == len(self.observed)
        lambda_ = matrix(self.hidden[j].phi)
        for i, o in enumerate(observed_values):
            assert self.observed[i].family._check_u(o)
            lambda_ += asmatrix(o) * self.W[i, j]
        return lambda_.A.reshape((self.hidden[j].family.size(),))


def contrastive_divergence_iteration(
        harmonium,
        observed_data,
        eta
):
    """
    One iteration of contrastive divergence
    """
    theta_update = [zeros_like(o.phi) for o in harmonium.observed]
    lambda_update = [zeros_like(h.phi) for h in harmonium.hidden]
    W_update = empty((len(theta_update), len(
        lambda_update)), dtype=numpy.object_)
    for i in xrange(len(theta_update)):
        for j in xrange(len(lambda_update)):
            W_update[i, j] = zeros(
                (len(theta_update[i]), len(lambda_update[j])))

    for d in observed_data:
        assert len(d) == len(harmonium.observed)

        # get parameters for hidden variables
        lambda_hat = [
            harmonium.lambda_hat(j, d)
            for j, h in enumerate(harmonium.hidden)
        ]

        # sample the hidden variables
        hidden_samples = [
            h.family.sample(phi)
            for h, phi in zip(harmonium.hidden, lambda_hat)
        ]

        # get parameters for observed variables
        theta_hat = [
            harmonium.theta_hat(i, hidden_samples)
            for i, o in enumerate(harmonium.observed)
        ]

        # sample the observed variables
        observed_samples = [
            o.family.sample(phi)
            for o, phi in zip(harmonium.observed, theta_hat)
        ]

        theta_update += d


if '__main__' == __name__:
    from infpy.exp import *
    discrete = DiscreteExpFamily(3)
    gaussian = GaussianExpFamily()
    h = Harmonium(
        observed=[Variable(discrete, discrete.phi([.5, .3, .2])), ],
        hidden=[Variable(gaussian, gaussian.phi([.0, 1.])), ]
    )
    observed_data = [[numpy.array([1, 0, 0])]]
    hidden_data = [[numpy.array([0.0, 0.0])]]
    #h.lambda_hat(0, observed_data[0])
    #h.theta_hat(0, hidden_data[0])
    contrastive_divergence_iteration(h, observed_data, 1e-3)
