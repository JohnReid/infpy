#
# Copyright John Reid 2006
#

"""
Code for variational inference in a dirichlet process mixture of exponential families model.

See U{http://www.cs.berkeley.edu/~jordan/papers/vdp-icml.pdf}.
"""

from numpy import array, zeros, ones, dot, empty, exp, log, where, outer, isfinite, empty_like
from scipy.special import digamma, gammaln
from scipy.stats import beta
from infpy.exp import DirichletExpFamily
#import infpy.exp; reload(infpy.exp)
from infpy.convergence_test import LlConvergenceTest
from functools import reduce

if __debug__:
    def debug_if_infinite(x):
        """Enter debugger if argument is not finite."""
        if not isfinite(x).all():
            import IPython
            IPython.Debugger.Pdb().set_trace()
            pass
else:
    def debug_if_infinite(x):
        pass


class VariationalDistribution(object):
    """
    The variational distribution, q, over the hidden variables.
    """

    _debug_LL = False
    "Whether to test if LL always increases across parameter updates."

    _do_order_updates = True
    "Whether we permute the components to keep the largest first."

    def __init__(self, K, X, alpha, _lambda, conj_prior):
        """
        Initialise the variational distribution
        @arg K: The truncation parameter: Upper limit on number of mixtures
        @arg X: The data
        @arg alpha: The dirichlet scaling parameter
        @arg _lambda: The prior for the etas
        @arg conj_prior: Conjugate prior to the exponential family that we are mixing
        family's log normalisation factor given the conjugate prior's natural parameters, eta
        """
        self.K = K
        "The truncation parameter: I.e. upper limit on # of mixtures"

        self.N = len(X)
        "The number of data"

        assert self.N == len(X), "Wrong number of data"
        assert self.N == 0 or conj_prior.likelihood._check_shape(
            X[0]), "Data dimensions wrong (should be as sufficient statistics)"
        self.X = X
        "The data"

        self.alpha = alpha
        "The dirichlet scaling parameter"

        self.conj_prior = conj_prior
        "Conjugate prior to the exponential family that we are mixing"

        self._lambda = _lambda
        assert self.conj_prior.prior._check_shape(_lambda)
        "The prior for the etas"

        self.d = conj_prior.strength_dimension
        "The number of dimensions in the conjugate prior that represent the strength of the prior"

        self.gamma = zeros((K - 1, 2))
        "Beta parameters for the distributions on V_i"

        self.tau = zeros((K, self.conj_prior.prior.dimension))
        "Natural parameters for distributions on eta_i"

        self.phi = zeros((self.N, K))
        "Multinomial parameters for distributions on Z_n"

        self._last_LL = log(0.0)
        "The last log likelihood calculated"

        self._randomise()

    def _randomise(self):
        """
        Randomise the variational parameters.
        """
        from numpy.random import dirichlet
        self.phi = dirichlet(ones(self.K), size=self.N)
        self.tau = outer(ones(self.K), self._lambda)
        self.gamma[:, 0] = 1.
        self.gamma[:, 1] = self.alpha

    def _update_order(self):
        """
        Reorders components to make largest first. Does not reorder gamma so _update_gamma should be
        called next in updating order.
        """
        expected_component_sizes = -self.phi.sum(axis=0)
        permutation = expected_component_sizes.argsort()
        new_phi = empty_like(self.phi)
        for n in range(self.N):
            for k in range(self.K):
                new_phi[n, k] = self.phi[n, permutation[k]]
        self.phi = new_phi
        new_tau = empty_like(self.tau)
        for k in range(self.K):
            new_tau[k] = self.tau[permutation[k]]
        self.tau = new_tau
        expected_component_sizes = self.phi.sum(axis=0)
        for k in range(self.K - 1):
            assert expected_component_sizes[k] >= expected_component_sizes[k + 1]

    def _update_gamma(self):
        "Update the gamma parameters."
        self.gamma[:, 0] = 1. + self.phi.sum(axis=0)[:-1]
        for i in range(self.K - 1):
            self.gamma[i, 1] = self.alpha + self.phi[:, i + 1:].sum()
        self._check_finite()

    def _update_tau(self):
        "Update the tau parameters."
        #import IPython; IPython.Debugger.Pdb().set_trace()
        for i in range(self.K):
            self.tau[i, :self.d] = self._lambda[:self.d] + self.phi[:, i].sum()
            self.tau[i, self.d:] = self._lambda[self.d:] + \
                sum(phi * x for x, phi in zip(self.X, self.phi[:, i]))
        self._check_finite()

    def _log_v_expectations(self):
        "The expectations of log V_i and log 1-V_i."
        second_term = digamma(self.gamma.sum(axis=1))
        #import IPython; IPython.Debugger.Pdb().set_trace()
        exp_log_v_i = empty(self.K)
        exp_log_one_minus_v_i = empty(self.K)
        exp_log_v_i[:-1] = digamma(self.gamma[:, 0]) - second_term
        exp_log_v_i[-1] = 0.
        exp_log_one_minus_v_i[:-1] = digamma(self.gamma[:, 1]) - second_term
        exp_log_one_minus_v_i[-1] = log(0.0)
        return exp_log_v_i, exp_log_one_minus_v_i

    def _update_phi(self):
        "Update the phi parameters."
        # calculate all we can before we visit each datum
        exp_log_v_i, exp_log_one_minus_v_i = self._log_v_expectations()
        sum_E_log_1_minus_Vj = array(
            [exp_log_one_minus_v_i[:i].sum() for i in range(self.K)])

        # expected values of the log normalisation factors
        exp_log_norm = array(
            [self.conj_prior.exp_likelihood_log_normalisation_factor(tau) for tau in self.tau])

        # expected value of the etas given the taus
        exp_eta = array([self.conj_prior.prior.exp_T(tau) for tau in self.tau])

        # the terms that don't depend on the X[n]s
        partial_E = exp_log_v_i + sum_E_log_1_minus_Vj - exp_log_norm

        # for each datum reestimate phi
        #import IPython; IPython.Debugger.Pdb().set_trace()
        for n in range(self.N):
            E = partial_E.copy() + dot(exp_eta[:, self.d:], self.X[n])
            E_exp = exp(E - E.max())  # scale to avoid numerical errors
            self.phi[n, :] = E_exp / E_exp.sum()
            # if not isfinite(self.phi).all():
            #               import IPython; IPython.Debugger.Pdb().set_trace()

        self._check_finite()

    def _check_finite(self):
        "Checks if the variational parameters are finite."
        assert isfinite(self.tau).all()
        assert isfinite(self.gamma).all()
        assert isfinite(self.phi).all()

    def update(self):
        """
        One iteration of variational updates.
        """
        last_LL = self._last_LL
        self._update_tau()
        if VariationalDistribution._debug_LL and self.log_likelihood() < last_LL:
            print('Tau update decreased LL by %.3f' % (last_LL - self._last_LL))

        last_LL = self._last_LL
        self._update_phi()
        if VariationalDistribution._debug_LL and self.log_likelihood() < last_LL:
            print('Phi update decreased LL by %.3f' % (last_LL - self._last_LL))

        if VariationalDistribution._do_order_updates:
            last_LL = self._last_LL
            self._update_order()
            if VariationalDistribution._debug_LL and self.log_likelihood() < last_LL:
                print('Order update decreased LL by %.3f' % (
                    last_LL - self._last_LL))

        last_LL = self._last_LL
        self._update_gamma()
        if VariationalDistribution._debug_LL and self.log_likelihood() < last_LL:
            print('Gamma update decreased LL by %.3f' % (
                last_LL - self._last_LL))

        # calculate LL if haven't done already for checks
        if not VariationalDistribution._debug_LL:
            self.log_likelihood()

        return self._last_LL

    def log_likelihood(self):
        """
        Returns the likelihood given the variational distribution. Eqn. 12 in
        U{http://www.cs.berkeley.edu/~jordan/papers/vdp-icml.pdf}.
        """
        exp_log_v_i, exp_log_one_minus_v_i = self._log_v_expectations()
        exp_eta = array([self.conj_prior.prior.exp_T(tau) for tau in self.tau])

        LL = 0.

        # KL for distributions over V
        dirichlet = DirichletExpFamily(k=2)
        alpha_eta = dirichlet.eta([self.alpha, 1.])
        # this if for the K'th V_i
        LL += gammaln(1. + self.alpha) - gammaln(self.alpha)
        for i in range(self.K - 1):  # for the first K-1 Vs
            eta = dirichlet.eta(self.gamma[i])
            LL -= dirichlet.KL(eta, alpha_eta)
        if __debug__:
            debug_if_infinite(LL)

        # KL for distributions over eta
        for i in range(self.K):
            LL -= self.conj_prior.prior.KL(self.tau[i], self._lambda)
        if __debug__:
            debug_if_infinite(LL)

        # LL for distributions over Z
        for n in range(self.N):
            for i in range(self.K - 1):
                LL += self.phi[n, i + 1:].sum() * exp_log_one_minus_v_i[i]
                LL += self.phi[n, i] * exp_log_v_i[i]
        if __debug__:
            debug_if_infinite(LL)

        # LL for x
        for n in range(self.N):
            for i in range(self.K):
                phi = self.phi[n, i]
                if 0.0 != phi:
                    LL += phi * (
                        dot(self.X[n], exp_eta[i, self.d:])
                        - self.conj_prior.exp_likelihood_log_normalisation_factor(self.tau[i])
                        + self.conj_prior.likelihood.h(self.X[n])
                        - log(phi)
                        # + exp_eta[i,:self.d].sum()
                    )
                    #import IPython; IPython.Debugger.Pdb().set_trace()
                    # pass
        if __debug__:
            debug_if_infinite(LL)

        self._last_LL = LL
        return LL

    def component_proportions(self):
        "@return: An array specifying the probability for a new point coming from each component."
        # chance of selecting this component given have not selected any of previous
        p_v_i_one = self.gamma[:, 0] / self.gamma.sum(axis=1)

        p_select_component = empty(self.K)  # result
        stick_left = 1.0  # amount of stick left in stick breaking representation of dirichlet process
        for k, q in enumerate(p_v_i_one):
            # chance of selecting this component
            p_select_component[k] = stick_left * q
            stick_left *= (1. - q)  # reduce amount of stick left
        p_select_component[-1] = stick_left
        if __debug__:
            debug_if_infinite(p_select_component)
        return p_select_component

    def predictive(self, x):
        """Predictive distribution of x given variational parameters."""

        # calculate chance of selecting components
        p_select_component = self.component_proportions()

        # calculate chance of data given each component parameters
        predictive_component = exp(
            [self.conj_prior.log_conjugate_predictive(
                x, self.tau[k]) for k in range(self.K)]
        )
        if __debug__:
            debug_if_infinite(predictive_component)

        #import IPython; IPython.Debugger.Pdb().set_trace()
        return (predictive_component * p_select_component).sum()


def polya_urn(N, alpha):
    """
    Draw N times from a Polya urn parameterised by alpha.

    @arg N: Number of draws to make.
    @arg alpha: Parameter for the polya urn. Small alphas result in a small number of components, big alphas
    spread the draws over more components.
    @return: A vector of counts. The i'th entry represents how many draws were made from the i'th component
    """
    from numpy.random import multinomial
    assert alpha >= 0.
    counts = []
    for n in range(N):
        k = len(counts)  # number of different samples so far
        multi_param = array([k == i and alpha or counts[i]
                             for i in range(k + 1)]) / (n + alpha)
        # get the index of the 1 in the multinomial sample
        sample = where(multinomial(1, multi_param))[0][0]
        # print multi_param, multi_sample, sample
        if len(counts) == sample:
            counts.append(1)  # we have drawn from a new component
        else:
            counts[sample] += 1  # we drew from an existing component
    return array(counts)


def generate_test_data(N, alpha, conjugate_prior, tau):
    """
    Generate N data points using given parameters.

    @return: (counts, eta, X)
    """

    # how many components are there and how many samples in each?
    counts = polya_urn(N, alpha)
    K = len(counts)

    # sample a natural parameter for each component
    eta = conjugate_prior.prior.sample(
        tau, size=K)[:, conjugate_prior.strength_dimension:]

    # sample the data
    X = empty((N, conjugate_prior.likelihood.dimension))
    n = 0
    for i in range(K):  # for each component
        X[n:n + counts[i]
          ] = conjugate_prior.likelihood.sample(eta[i], size=counts[i])
        n += counts[i]

    return counts, eta, X


if '__main__' == __name__:
    from infpy.exp import NormalWishartExpFamily, MvnExpFamily, MvnConjugatePrior, MultinomialConjugatePrior
    from pylab import figure, plot, arange, close, meshgrid, axes, bar, title, contour, contourf, imshow
    from numpy.linalg import inv, eig

    class DpTestCase(object):
        markers = ['+', 'o', '1', '2', '3', '4',
                   'd', 's', 'v', 'x', '>', '<', ',', '.']
        scatter_markers = ['s', 'o', '^', '>', 'v', '<', 'd', 'p', 'h', '8']
        colours = ['r', 'g', 'b', 'y', 'm', 'c', 'k', 'w']

        def __init__(self, N, alpha, K):
            self.N = N
            self.alpha = alpha
            self.K = K

        def generate_test_data(self):
            self.counts, self.eta, self.X = generate_test_data(
                self.N, self.alpha, self.conj_prior, self.tau)
            print('%d data comes from %d components partitioned as %s' % (
                len(self.X), len(self.counts), str(self.counts)))
            self.components = []  # the components for each datum
            for i, c in enumerate(self.counts):
                self.components.extend([i] * c)

        def create_variational_distribution(self):
            self.var_dist = VariationalDistribution(
                self.K,
                self.X,
                self.alpha,
                self.tau,
                self.conj_prior,
                self.conj_prior.exp_likelihood_log_normalisation_factor
            )

        def plot(self):
            print('LL: %f' % self.var_dist.log_likelihood())

    class MvnTestCase(DpTestCase):
        def __init__(self, N, alpha, K):
            DpTestCase.__init__(self, N, alpha, K)
            self.conj_prior = MvnConjugatePrior(2)
            self.nu = 3.
            self.S = array([[1., 0.], [0., 1.]])
            self.kappa_0 = .01
            self.mu_0 = array([0., 0.])
            self.tau = self.conj_prior.prior.eta(
                (self.nu, self.S, self.kappa_0, self.mu_0))

        def generate_test_data(self):
            DpTestCase.generate_test_data(self)
            self.X_2d = array([self.conj_prior.likelihood.x(x)
                               for x in self.X])

        def var_dist_info(self):
            mus = []
            for i in range(self.var_dist.K):
                nu, S, kappa_0, mu_0 = self.var_dist.conj_prior.prior.theta(
                    self.var_dist.tau[i])
                #print 'mu_%d: %s' % (i, str(mu_0))
                #plot([mu_0[0]], [mu_0[1]], marker='d', markersize=5, color='w')
                mus.append(mu_0)
                sigma = inv(nu * kappa_0 * S)
                w, v = eig(sigma)  # plot the covariance
                for j in range(2):
                    evec = w[j] * v[:, j]
                    plot([mu_0[0] - evec[0], mu_0[0] + evec[0]], [mu_0[1] -
                                                                  evec[1], mu_0[1] + evec[1]], color=DpTestCase.colours[i])

            p_v = self.var_dist.gamma[:, 0] / self.var_dist.gamma.sum(axis=1)
            stick_lengths = zeros((self.var_dist.K))
            for i in range(self.var_dist.K - 1):
                stick_lengths[i] = p_v[i] * \
                    reduce(float.__mul__, (1. - p_v[j] for j in range(i)), 1.)
            stick_lengths[-1] = 1. - stick_lengths.sum()
            #print 'Stick lengths: %s' % str(stick_lengths)
            #import IPython; IPython.Debugger.Pdb().set_trace()
            # pass
            return array(mus)

        def plot_data(self):
            assert len(DpTestCase.scatter_markers) >= len(self.counts)
            i = 0
            for component, c in enumerate(self.counts):
                #scatter(self.X_2d[i:i+c,0], self.X_2d[i:i+c,1], c=DpTestCase.colours[component] * c)
                scatter(self.X_2d[i:i + c, 0], self.X_2d[i:i + c, 1],
                        c='w', marker=DpTestCase.scatter_markers[component])
                i += c

        def plot_phi(self):
            a = axes([.9, .9, .1, .1], axisbg='w')
            rects = bar(range(self.var_dist.K), self.var_dist.phi.sum(
                axis=0), color=DpTestCase.colours[:self.var_dist.K])
            title("Components")
            setp(a, xticks=[], yticks=[])

        def plot_contours(self, steps=12):
            xmin, xmax, ymin, ymax = axis()
            xstep = (xmax - xmin) / steps
            ystep = (ymax - ymin) / steps
            range0 = arange(xmin, xmax + xstep / 100., step=xstep)
            range1 = arange(ymin, ymax + ystep / 100., step=ystep)
            mesh0, mesh1 = meshgrid(range0, range1)
            z = empty(mesh0.shape)
            print(xstep, ystep, range0[-1], range1[-1])
            for i0 in range(len(z)):
                for i1 in range(len(z[0])):
                    T = self.conj_prior.likelihood.T(
                        array([mesh0[i0, i1], mesh1[i0, i1]]))
                    z[i0, i1] = self.var_dist.predictive(T)
            contour(z, cmap=cm.gray_r, extent=axis())
            #imshow(z, cmap=cm.gray)

        def plot(self):
            DpTestCase.plot(self)
            figure()
            self.mus = self.var_dist_info()
            self.plot_data()
            # self.plot_contours()
            self.plot_phi()

    class MvnMultiDTestCase(DpTestCase):
        def __init__(self, N, alpha, K, dims=4):
            DpTestCase.__init__(self, N, alpha, K)
            self.conj_prior = MvnConjugatePrior(dims)
            self.nu = dims + 1
            self.S = identity(dims)
            self.kappa_0 = .01
            self.mu_0 = zeros((dims,))
            self.tau = self.conj_prior.prior.eta(
                (self.nu, self.S, self.kappa_0, self.mu_0))

        def generate_test_data(self):
            DpTestCase.generate_test_data(self)
            self.X_2d = array([self.conj_prior.likelihood.x(x)
                               for x in self.X])

        def var_dist_info(self):
            mus = []
            for i in range(self.var_dist.K):
                nu, S, kappa_0, mu_0 = self.var_dist.conj_prior.prior.theta(
                    self.var_dist.tau[i])
                #print 'mu_%d: %s' % (i, str(mu_0))
                #plot([mu_0[0]], [mu_0[1]], marker='d', markersize=5, color='w')
                mus.append(mu_0)
                sigma = inv(nu * kappa_0 * S)
                w, v = eig(sigma)  # plot the covariance
                for j in range(2):
                    evec = w[j] * v[:, j]
                    plot([mu_0[0] - evec[0], mu_0[0] + evec[0]], [mu_0[1] -
                                                                  evec[1], mu_0[1] + evec[1]], color=DpTestCase.colours[i])

            p_v = self.var_dist.gamma[:, 0] / self.var_dist.gamma.sum(axis=1)
            stick_lengths = zeros((self.var_dist.K))
            for i in range(self.var_dist.K - 1):
                stick_lengths[i] = p_v[i] * \
                    reduce(float.__mul__, (1. - p_v[j] for j in range(i)), 1.)
            stick_lengths[-1] = 1. - stick_lengths.sum()
            #print 'Stick lengths: %s' % str(stick_lengths)
            #import IPython; IPython.Debugger.Pdb().set_trace()
            # pass
            return array(mus)

        def plot(self):
            DpTestCase.plot(self)

    class MultinomialTestCase(DpTestCase):
        def __init__(self, N, alpha, K):
            DpTestCase.__init__(self, N, alpha, K)
            self.dims = 3
            self.conj_prior = MultinomialConjugatePrior(k=self.dims)
            self.tau = self.conj_prior.prior.eta(ones(self.dims))

        def generate_test_data(self):
            DpTestCase.generate_test_data(self)
            self.samples = array(
                [self.conj_prior.likelihood.x(x) for x in self.X])

        def plot(self):
            DpTestCase.plot(self)

    close('all')
    # from numpy.random import seed; seed(1)
    # test_case = MvnMultiDTestCase(N=40, alpha=1., K=4, dims=37)
    test_case = MvnTestCase(N=40, alpha=1., K=4)
    # test_case = MultinomialTestCase(N=40, alpha=1., K=4)
    test_case.generate_test_data()
    test_case.create_variational_distribution()
    test_case.plot()
    convergence_test = LlConvergenceTest()
    from time import time
    start = time()
    max_iters = 30
    for i in range(max_iters):
        LL = test_case.var_dist.update()
        test_case.plot()
        if convergence_test(LL):
            break
    total_elapsed = time() - start
    print('%d iterations took %f secs, %f secs/iteration' % (
        i + 1, total_elapsed, total_elapsed / i + 1))
