#
# Copyright John Reid 2010
#


"""
Implementation of HDPM with uncertainty in the words.
"""

import logging
import numpy
import numpy.random as R
from numpy import log, exp
from scipy.stats import bernoulli, poisson
from scipy.special import digamma, polygamma, gammaln
from optparse import OptionGroup, OptionParser
from cookbook.named_tuple import namedtuple
from cookbook.dicts import DictOf
from ..math import GammaDist, BetaDist, DirichletDist, discrete_sample, _safe_x_log_x, array_replace
from ..math import approximate_expected_num_tables, approximate_beta_expectation
from ..math import calculate_P_plus, _permute, _permutation_by_sort, _greater_than, _approx_f_n


def add_hyper_parameter_options(parser):
    "Add the hyper-parameter options to the parser."
    group = OptionGroup(
        parser,
        "Hyper-parameter options",
        "Options to control the hyper-parameters of the HDPM."
    )
    for p in ('alpha', 'beta', 'gamma', 'lambda'):
        parser.add_option(
            "--a_%s" % p,
            dest="a_%s" % p,
            type='float',
            default=4.
        )
        parser.add_option(
            "--b_%s" % p,
            dest="b_%s" % p,
            type='float',
            default=4.
        )
    parser.add_option(
        "--a_tau",
        dest="a_tau",
        default=None
    )
    parser.add_option(
        "--a_omega",
        dest="a_omega",
        default=None
    )
    parser.add_option(
        "--posterior-enrichment-threshold",
        dest="posterior_enrichment_threshold",
        default=2.,
        type='float',
        help='Threshold for associations with TPs in the posterior.'
    )
    parser.add_option(
        "--topic-size-threshold",
        dest="topic_size_threshold",
        default=1.,
        type='float',
        help='Threshold for how big a topic is to be counted.'
    )
    parser.add_option_group(group)


def parse_options(parser):
    options, args = parser.parse_args()
    logging.info('Options:')
    for option in parser.option_list:
        if option.dest:
            logging.info('%32s: %-32s * %s', option.dest,
                         str(getattr(options, option.dest)), option.help)
    return options, args


def get_default_options():
    "@return: The default HDPM options."
    parser = OptionParser()
    add_hyper_parameter_options(parser)
    return parser.get_default_values()


class Data(object):
    """
    The data for a HDPM model with uncertainty in the binding sites.
    """

    def __init__(
        self,
        genes,
        F,
        options
    ):
        "Construct. Each gene should be a list of (factor, list of rho_gi)"

        self.genes = genes
        "The data"

        self.n_g = numpy.array(
            [
                sum(len(rho) for _f, rho in g)
                for g in self.genes
            ]
        )
        "Number of putative sites in each gene."

        self.N = self.n_g.sum()
        "Total number of putative sites."

        self.F = F
        "The number of factors in our data."

        self.options = options
        "Options (including the hyper-parameters)."

        self.G = len(self.genes)
        "Number of genes."

        # check hyper-parameters
        if None == options.a_tau:
            logging.debug('a_tau not specified, using uniform distribution.')
            options.a_tau = numpy.ones(self.F)
        if None == options.a_omega:
            logging.debug('a_omega not specified, using uniform distribution.')
            options.a_omega = numpy.ones(self.F)

    def _check_shapes(self):
        assert self.G == len(self.genes)
        assert self.G == len(self.n_g)


class CountTrio(object):
    """
    Holds matrices to store E, V, and Z for random counts.
    """

    def __init__(self, shape):
        self.E = numpy.zeros(shape)
        self.V = numpy.zeros(shape)
        self.Z = numpy.zeros(shape)

    def _check_shapes(self, shape):
        assert shape == self.E.shape
        assert shape == self.V.shape
        assert shape == self.Z.shape

    def _check_E_V_finite(self):
        assert numpy.isfinite(self.E).all()
        assert numpy.isfinite(self.V).all()

    def sum(self, *args, **kfargs):
        E = self.E.sum(*args, **kfargs)
        V = self.V.sum(*args, **kfargs)
        Z = self.Z.sum(*args, **kfargs)
        result = CountTrio(E.shape)
        result.E = E
        result.V = V
        result.Z = Z
        return result

    def calculate_P_plus(self):
        self.P_plus, self.E_plus, self.V_plus = calculate_P_plus(
            self.E, self.V, self.Z)

    def permute(self, permutation, axis):
        "Permute the counts by the permutation on the given axis."
        self.E = _permute(self.E, permutation, axis=axis)
        self.V = _permute(self.V, permutation, axis=axis)
        self.Z = _permute(self.Z, permutation, axis=axis)
        if hasattr(self, 'P_plus'):
            self.P_plus = _permute(self.P_plus, permutation, axis=axis)
            self.E_plus = _permute(self.E_plus, permutation, axis=axis)
            self.V_plus = _permute(self.V_plus, permutation, axis=axis)


class ApproxCounts(object):
    "Holds approximate counts that are recalculated after estimates of z and v have been updated."

    def __init__(self, var_dist):
        "Construct."
        self._n_gkf = CountTrio(
            (var_dist.data.G, var_dist.K + 1, var_dist.data.F))
        self.recalculate(var_dist)

    def _check_shapes(self, shape):
        "Check the shapes of the arrays are correct."
        self._n_gkf._check_shapes(shape)
        self.n_gk._check_shapes(shape[:2])
        self.n_kf._check_shapes(shape[1:])
        self.n_g._check_shapes((shape[0],))
        self.n_k._check_shapes((shape[1],))

    def _check_finite(self):
        "Check the counts are finite."
        self._n_gkf._check_finite()
        self.n_gkf._check_finite()
        self.n_gk._check_finite()
        self.n_kf._check_finite()
        self.n_g._check_finite()
        self.n_k._check_finite()

    def recalculate(self, var_dist):
        data = var_dist.data

        # for each gene
        for g, (gene, d_g) in enumerate(zip(data.genes, var_dist.d)):
            for (f, _), d_f in zip(gene, d_g):
                not_d_f = 1. - d_f
                self._n_gkf.E[g, :, f] += d_f.sum(axis=0)
                self._n_gkf.V[g, :, f] += (d_f * not_d_f).sum(axis=0)
                self._n_gkf.Z[g, :, f] += log(not_d_f).sum(axis=0)
        self._n_gkf._check_E_V_finite()

        self.n_gk = self._n_gkf.sum(axis=2)
        self.n_kf = self._n_gkf.sum(axis=0)
        self.n_g = self.n_gk.sum(axis=1)
        self.n_k = self.n_kf.sum(axis=1)

        self.n_gk.calculate_P_plus()
        self.n_kf.calculate_P_plus()
        self.n_g.calculate_P_plus()
        self.n_k.calculate_P_plus()

    def __str__(self):
        return 'ApproxCounts object'

    def permute(self, permutation):
        "Permute the counts based on a permutation of the programs."
        self.n_gk.permute(permutation, axis=1)
        self.n_kf.permute(permutation, axis=0)
        self.n_k.permute(permutation, axis=0)
        # don't bother permuting self._n_gkf as we only use it internally


def approx_log_a_plus_n(a, E, V):
    "@return: An approximation to log(a+n)."
    term = a + E
    return log(term) - V / (2. * (term)**2)


def sample_rho(G, average_n_g):
    "Sample random values of rho."
    from numpy.random import rand
    return [rand(n_g) for n_g in poisson.rvs(average_n_g, size=G)]


def sample(hyperparameters, rho, K, F):
    "Sample from the model."
    G = len(rho)

    q_alpha = GammaDist(hyperparameters.a_alpha, hyperparameters.b_alpha)
    alpha = q_alpha.sample()

    q_beta = GammaDist(hyperparameters.a_beta, hyperparameters.b_beta)
    beta = q_beta.sample()

    q_gamma = GammaDist(hyperparameters.a_gamma, hyperparameters.b_gamma)
    gamma = q_gamma.sample()

    q_lambda = GammaDist(hyperparameters.a_lambda, hyperparameters.b_lambda)
    lambda_ = q_lambda.sample()

    q_tau = DirichletDist(hyperparameters.a_tau)
    tau = q_tau.sample()

    q_omega = DirichletDist(hyperparameters.a_omega)
    omega = q_omega.sample()

    q_pi_bar = BetaDist(numpy.ones(K), gamma * numpy.ones(K))
    pi_bar = q_pi_bar.sample()

    pi = numpy.empty_like(pi_bar)
    for k in range(K - 1):
        pi[k] = pi_bar[k] * (1. - pi_bar[:k]).prod()
    pi[-1] = 1. - pi[:-1].sum()
    if pi[-1] < 0.:  # adjust for numerical errors
        pi[-1] = 0.

    theta = numpy.random.dirichlet(alpha * pi, size=G)
    phi = numpy.empty((K + 1, F))
    phi[0] = numpy.random.dirichlet(lambda_ * omega)
    phi[1:] = numpy.random.dirichlet(beta * tau, size=K)

    # sample the correct number of sites for each gene
    sites = [None] * G
    for g, rho_g in enumerate(rho):
        v_g = [bernoulli(rho_i) for rho_i in rho_g]
        z_g = [v_gi and discrete_sample(theta[g]) + 1 or 0 for v_gi in v_g]
        x_g = [discrete_sample(phi[z_gi]) for z_gi in z_g]
        sites[g] = (v_g, z_g, x_g)

    result_type = namedtuple(
        'Sample', 'alpha beta gamma lambda_ tau omega pi_bar pi theta phi sites')

    return result_type(
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        lambda_=lambda_,
        tau=tau,
        omega=omega,
        pi_bar=pi_bar,
        pi=pi,
        theta=theta,
        phi=phi,
        sites=sites,
    )


def genes_from_sites(sites, rho):
    "Convert samples sites to data structure ready to pass to class Data."
    result = []
    for (_v_g, _z_g, x_g), rho_g in zip(sites, rho):
        _tmp = DictOf(list)
        for x_gi, rho_gi in zip(x_g, rho_g):
            _tmp[x_gi].append(rho_gi)
        result.append(list(_tmp.items()))
    return result


class VariationalDistribution(object):
    """
    A variational distribution over the hidden variables
    """

    def __init__(self, data, K):
        "Construct."

        self.data = data
        "Data for model."

        self.K = K
        "Maximum number of topics."

        self._pre_calculate()
        self._initialise_dists()
        self._calculate_counts()
        self._check_shapes()

    def _pre_calculate(self):
        "Pre-calculate some values we will use repeatedly."
        self.log_rho = [None] * self.data.G
        self.log_1_minus_rho = [None] * self.data.G
        for g, gene in enumerate(self.data.genes):
            self.log_rho[g] = [None] * len(gene)
            self.log_1_minus_rho[g] = [None] * len(gene)
            for i, (_, rhos) in enumerate(gene):
                self.log_rho[g][i] = numpy.outer(log(rhos), numpy.ones(self.K))
                self.log_1_minus_rho[g][i] = log(1. - numpy.asarray(rhos))

    def _check_shapes(self):
        "Check the shapes of arrays."
        if __debug__:
            G, K, F = self.data.G, self.K, self.data.F

            self.data._check_shapes()

            assert (G,) == self.E_log_eta.shape
            assert (K + 1,) == self.E_log_xi.shape
            assert (G, K) == self.E_s_gk.shape
            assert (K + 1, F) == self.E_t_kf.shape

            assert (F,) == self.q_tau.G.shape
            assert (F,) == self.q_omega.G.shape
            assert (K,) == self.G_pi.shape

            assert len(self.d) == G
            for d_g, log_rho_g, log_1_minus_rho_g in zip(self.d, self.log_rho, self.log_1_minus_rho):
                for d_gf, log_rho_gf, log_1_minus_rho_gf in zip(d_g, log_rho_g, log_1_minus_rho_g):
                    n = len(log_1_minus_rho_gf)
                    assert (n,) == log_1_minus_rho_gf.shape
                    assert (n, K) == log_rho_gf.shape
                    assert (n, K + 1) == d_gf.shape

            self.counts._check_shapes((G, K + 1, F))

    def _initialise_dists(self):
        """
        Initialise the variational distributions of a HDPM by sampling the generative model (ignoring the data).
        """
        from numpy.random import rand

        self.q_gamma = GammaDist(
            self.data.options.a_gamma, self.data.options.b_gamma)
        "Variational distribution over gamma."
        gamma = self.q_gamma.sample()

        self.q_alpha = GammaDist(
            self.data.options.a_alpha, self.data.options.b_alpha)
        "Variational distribution over alpha."
        alpha = self.q_alpha.sample()

        self.q_beta = GammaDist(self.data.options.a_beta,
                                self.data.options.b_beta)
        "Variational distribution over beta."

        self.q_gamma = GammaDist(
            self.data.options.a_gamma, self.data.options.b_gamma)
        "Variational distribution over gamma."

        self.q_lambda = GammaDist(
            self.data.options.a_lambda, self.data.options.b_lambda)
        "Variational distribution over lambda."

        self.q_tau = DirichletDist(self.data.options.a_tau)
        "Variational distribution over tau."

        self.q_omega = DirichletDist(self.data.options.a_omega)
        "Variational distribution over omega."

        self.q_pi_bar = BetaDist(numpy.ones(
            self.K), gamma * numpy.ones(self.K))
        "Variational distribution over pi_bar"

        #
        # do some sampling
        #
        pi_bar = self.q_pi_bar.sample()
        pi = numpy.empty_like(pi_bar)
        for k in range(self.K - 1):
            pi[k] = pi_bar[k] * (1. - pi_bar[:k]).prod()
        pi[-1] = 1. - pi[:-1].sum()
        if pi[-1] < 0.:  # adjust for numerical errors
            pi[-1] = 0.

        alpha_pi = alpha * pi
        theta = numpy.random.dirichlet(alpha_pi, size=self.data.G)

        self.d = [None] * self.data.G
        "Distribution over c and d."
        for g, (t, gene) in enumerate(zip(theta, self.data.genes)):
            self.d[g] = [None] * len(gene)
            for i, (_, rhos) in enumerate(gene):
                self.d[g][i] = numpy.zeros((len(rhos), self.K + 1))
                for j, rho in enumerate(rhos):
                    if rand() < rho:
                        self.d[g][i][j, 0] = 1.
                    else:
                        self.d[g][i][j, discrete_sample(t) + 1] = 1.

        self._calculate_counts()
        self._calculate_G_pi()
        self._calculate_E_s_gk()
        self._calculate_E_t_kf()
        self._calculate_E_log_xi()
        self._calculate_E_log_eta()

        return gamma, alpha, pi_bar, pi, theta

    def _calculate_E_s_gk(self):
        self.E_s_gk = approximate_expected_num_tables(
            self.q_alpha.G * self.G_pi,
            self.counts.n_gk.P_plus[:, 1:],
            self.counts.n_gk.E_plus[:, 1:],
            self.counts.n_gk.V_plus[:, 1:],
        )

    def _calculate_E_t_kf(self):
        self.E_t_kf = numpy.empty((self.K + 1, self.data.F))
        self.E_t_kf[0] = approximate_expected_num_tables(
            self.q_lambda.G * self.q_omega.G,
            self.counts.n_kf.P_plus[0],
            self.counts.n_kf.E_plus[0],
            self.counts.n_kf.V_plus[0],
        )
        self.E_t_kf[1:] = approximate_expected_num_tables(
            self.q_beta.G * self.q_tau.G,
            self.counts.n_kf.P_plus[1:],
            self.counts.n_kf.E_plus[1:],
            self.counts.n_kf.V_plus[1:],
        )

    def _calculate_E_log_eta(self):
        self.E_log_eta = approximate_beta_expectation(
            self.q_alpha.E,
            self.counts.n_g.P_plus,
            self.counts.n_g.E_plus,
            self.counts.n_g.V_plus,
        )

    def _calculate_E_log_xi(self):
        self.E_log_xi = numpy.empty((self.K + 1,))
        self.E_log_xi[0] = approximate_beta_expectation(
            self.q_lambda.E,
            self.counts.n_k.P_plus[0],
            self.counts.n_k.E_plus[0],
            self.counts.n_k.V_plus[0],
        )
        self.E_log_xi[1:] = approximate_beta_expectation(
            self.q_beta.E,
            self.counts.n_k.P_plus[1:],
            self.counts.n_k.E_plus[1:],
            self.counts.n_k.V_plus[1:],
        )

    def _calculate_G_pi(self):
        accum = numpy.empty((self.K,))
        accum[0] = 1.
        for k in range(1, self.K):
            accum[k] = accum[k - 1] * self.q_1_minus_pi_bar.G[k - 1]
        self.G_pi = self.q_pi_bar.G * accum
        assert numpy.isfinite(self.G_pi).all()
        assert self.G_pi.sum() > 0.
        return self.G_pi

    def _calculate_counts(self):
        self.counts = ApproxCounts(self)

    def _set_q_pi_bar(self, q_pi_bar):
        self._q_pi_bar = q_pi_bar
        # the distributions for (1-pi_bar)
        self._q_1_minus_pi_bar = BetaDist(self.q_pi_bar.b, self.q_pi_bar.a)

    def _update_alpha(self):
        self.q_alpha = GammaDist(
            self.data.options.a_alpha + self.E_s_gk.sum(),
            self.data.options.b_alpha - self.E_log_eta.sum()
        )

    def _update_beta(self):
        self.q_beta = GammaDist(
            self.data.options.a_beta + self.E_t_kf[1:].sum(),
            self.data.options.b_beta - self.E_log_xi[1:].sum()
        )

    def _update_lambda(self):
        self.q_lambda = GammaDist(
            self.data.options.a_lambda + self.E_t_kf[0].sum(),
            self.data.options.b_lambda - self.E_log_xi[0]
        )

    def _update_pi_bar(self):
        E_s_k = self.E_s_gk.sum(axis=0)
        assert E_s_k.shape == (self.K,)
        E_s_greater_k = _greater_than(E_s_k)
        self.q_pi_bar = BetaDist(E_s_k + 1., self.q_gamma.E + E_s_greater_k)
        self._calculate_G_pi()
        self._calculate_E_s_gk()

    def _update_gamma(self):
        self.q_gamma = GammaDist(
            self.data.options.a_gamma + self.K,
            self.data.options.b_gamma - log(self.q_1_minus_pi_bar.G).sum()
        )

    def _update_tau(self):
        self.q_tau = DirichletDist(
            self.data.options.a_tau + self.E_t_kf[1:].sum(axis=0))

    def _update_omega(self):
        self.q_omega = DirichletDist(
            self.data.options.a_omega + self.E_t_kf[0])

    def _update_v_z(self):
        for g, (x_g, d_g, log_rho_g, log_1_minus_rho_g) in enumerate(zip(self.data.genes, self.d, self.log_rho, self.log_1_minus_rho)):
            for (f, _rho), d_gf, log_rho_gf, log_1_minus_rho_gf in zip(x_g, d_g, log_rho_g, log_1_minus_rho_g):
                a_gf = numpy.empty_like(d_gf)
                V = (1. - d_gf) * d_gf
                a_gf[:, 0] = (
                    log_1_minus_rho_gf
                    - approx_log_a_plus_n(
                        self.q_lambda.E,
                        self.counts.n_k.E[0] - d_gf[:, 0],
                        self.counts.n_k.V[0] - V[:, 0]
                    )
                    + approx_log_a_plus_n(
                        self.q_lambda.G * self.q_omega.G[f],
                        self.counts.n_kf.E[0, f] - d_gf[:, 0],
                        self.counts.n_kf.V[0, f] - V[:, 0]
                    )
                )
                a_gf[:, 1:] = (
                    log_rho_gf
                    - approx_log_a_plus_n(
                        self.q_alpha.E,
                        self.counts.n_g.E[g] -
                        self.counts.n_gk.E[g, 0] - d_gf[:, 1:],
                        self.counts.n_g.V[g] -
                        self.counts.n_gk.V[g, 0] - V[:, 1:]
                    )
                    + approx_log_a_plus_n(
                        self.q_alpha.G * self.G_pi,
                        self.counts.n_gk.E[g, 1:] - d_gf[:, 1:],
                        self.counts.n_gk.V[g, 1:] - V[:, 1:]
                    )
                    - approx_log_a_plus_n(
                        self.q_beta.E,
                        self.counts.n_k.E[1:] - d_gf[:, 1:],
                        self.counts.n_k.V[1:] - V[:, 1:]
                    )
                    + approx_log_a_plus_n(
                        self.q_beta.G * self.q_tau.G[f],
                        self.counts.n_kf.E[1:, f] - d_gf[:, 1:],
                        self.counts.n_kf.V[1:, f] - V[:, 1:]
                    )
                )
                exp_a = exp(a_gf)  # exponentiate
                # normalise exponentiated a
                d_gf[:] = (exp_a.T / exp_a.sum(axis=1)).T
                array_replace(d_gf, numpy.isnan(d_gf), 1.)
                if __debug__:
                    _sum = d_gf.sum(axis=1)
                    assert (.99 < _sum).all()
                    assert (_sum < 1.01).all()
        self._calculate_counts()
        self._reorder_topic_labels()
        self._calculate_E_s_gk()
        self._calculate_E_t_kf()
        self._calculate_E_log_xi()
        self._calculate_E_log_eta()

    def _reorder_topic_labels(self):
        "Reorder the topic labels so that the largest topic is always first."

        self._check_shapes()

        # calculate the permutation required to put the topics in order of size.
        permutation = _permutation_by_sort(self.counts.n_k.E[1:])

        # don't bother re-ordering anything if permutation is the identity
        if (numpy.arange(self.K) != permutation).any():

            # create a new permutation that leaves k=0 where it is
            new_permutation = numpy.empty(
                (self.K + 1), dtype=permutation.dtype)
            new_permutation[0] = 0
            new_permutation[1:] = permutation + 1

            #start_LL = numpy.array(self._log_likelihood())

            for d_gene in self.d:
                for d_f in d_gene:
                    d_f[:] = _permute(d_f, new_permutation, axis=1)

            a_permuted = _permute(self.q_pi_bar.a, permutation, axis=0)
            b_permuted = _permute(self.q_pi_bar.b, permutation, axis=0)
            self.q_pi_bar.a = a_permuted
            self.q_pi_bar.b = b_permuted
            self.q_1_minus_pi_bar.a = b_permuted
            self.q_1_minus_pi_bar.b = a_permuted

            # permute the cached values (they all depend on the topic ordering)
            self.E_s_gk = _permute(self.E_s_gk, permutation, axis=1)
            self.E_t_kf = _permute(self.E_t_kf, new_permutation, axis=0)
            self.E_log_xi = _permute(self.E_log_xi, new_permutation, axis=0)
            self.G_pi = _permute(self.G_pi, permutation, axis=0)
            self.counts.permute(new_permutation)

            # check topics are sorted in decreasing order of size
            assert (self.counts.n_k.E[1:-1] >= self.counts.n_k.E[2:]).all()

    def update(self):
        "Update the model. That is do one iteration of learning."
        for update_fn in VariationalDistribution.update_fns:
            update_fn(self)
        self._check_shapes()

    update_fns = (
        _update_v_z,
        _update_pi_bar,
        _update_alpha,
        _update_beta,
        _update_lambda,
        _update_gamma,
        _update_tau,
        _update_omega,
    )
    """
    The update functions for the HDPM. This list determines the order in which the variational
    distributions are updated.
    """

    q_pi_bar = property(
        lambda self: self._q_pi_bar,
        _set_q_pi_bar,
        None,
        "Variational distribution over pi bar."
    )

    q_1_minus_pi_bar = property(
        lambda self: self._q_1_minus_pi_bar,
        None,
        None,
        "Variational distribution over 1 minus pi bar."
    )

    def _log_likelihood(self):
        """
        Returns the variational bound on the log likelihood.
        """
        LL = []  # build a list of all the different terms in the LL
        # we do this so we can examine which ones change after particular variational
        # updates

        # rho, v term
        term = 0.
        for d_g, log_rho_g, log_1_minus_rho_g in zip(self.d, self.log_rho, self.log_1_minus_rho):
            for d_gf, log_rho_gf, log_1_minus_rho_gf in zip(d_g, log_rho_g, log_1_minus_rho_g):
                term += (d_gf[:, 0] * log_1_minus_rho_gf).sum()
                term += ((1. - d_gf[:, 0]) * log_rho_gf[:, 0]).sum()
        assert numpy.isfinite(term)
        LL.append(term)

        # n_g term
        E_alpha = self.q_alpha.E
        log_gamma_E_alpha = gammaln(E_alpha)

        def f(n): return log_gamma_E_alpha - gammaln(E_alpha + n)

        def f2(n): return -polygamma(1, E_alpha + n)
        P_plus_n_g_k_positive = 1. - \
            reduce(numpy.ndarray.__mul__,
                   (1. - self.counts.n_gk.P_plus[:, 1:]).T).T
        term = _approx_f_n(
            f,
            f2,
            P_plus_n_g_k_positive,
            self.data.n_g - self.counts.n_gk.E_plus[:, 0],
            self.counts.n_gk.V_plus[:, 0]
        ).sum()
        assert numpy.isfinite(term)
        LL.append(term)

        # n_gk term
        G_alpha_pi = self.G_pi * self.q_alpha.G
        log_gamma_G_alpha_pi = gammaln(G_alpha_pi)

        def f(n): return gammaln(G_alpha_pi + n) - log_gamma_G_alpha_pi

        def f2(n): return polygamma(1, G_alpha_pi + n)
        term = _approx_f_n(
            f,
            f2,
            self.counts.n_gk.P_plus[:, 1:],
            self.counts.n_gk.E_plus[:, 1:],
            self.counts.n_gk.V_plus[:, 1:],
        ).sum()
        assert numpy.isfinite(term)
        LL.append(term)

        # n_k term
        E_beta = self.q_beta.E
        log_gamma_E_beta = gammaln(E_beta)

        def f(n): return log_gamma_E_beta - gammaln(E_beta + n)

        def f2(n): return -polygamma(1, E_beta + n)
        term = _approx_f_n(
            f,
            f2,
            self.counts.n_k.P_plus[1:],
            self.counts.n_k.E_plus[1:],
            self.counts.n_k.V_plus[1:]
        ).sum()
        assert numpy.isfinite(term)
        LL.append(term)

        # n_kf term
        G_beta_tau = self.q_beta.G * self.q_tau.G
        log_gamma_G_beta_tau = gammaln(G_beta_tau)

        def f(n): return gammaln(G_beta_tau + n) - log_gamma_G_beta_tau

        def f2(n): return polygamma(1, G_beta_tau + n)
        term = _approx_f_n(
            f,
            f2,
            self.counts.n_kf.P_plus[1:],
            self.counts.n_kf.E_plus[1:],
            self.counts.n_kf.V_plus[1:],
        ).sum()
        assert numpy.isfinite(term)
        LL.append(term)

        # n_k=0 term
        E_lambda = self.q_lambda.E
        log_gamma_E_lambda = gammaln(E_lambda)

        def f(n): return log_gamma_E_lambda - gammaln(E_lambda + n)

        def f2(n): return -polygamma(1, E_lambda + n)
        term = _approx_f_n(
            f,
            f2,
            self.counts.n_k.P_plus[0],
            self.counts.n_k.E_plus[0],
            self.counts.n_k.V_plus[0]
        ).sum()
        assert numpy.isfinite(term)
        LL.append(term)

        # n_0f term
        G_lambda_omega = self.q_lambda.G * self.q_omega.G
        log_gamma_G_lambda_omega = gammaln(G_lambda_omega)

        def f(n): return gammaln(G_lambda_omega + n) - log_gamma_G_lambda_omega

        def f2(n): return polygamma(1, G_lambda_omega + n)
        term = _approx_f_n(
            f,
            f2,
            self.counts.n_kf.P_plus[0],
            self.counts.n_kf.E_plus[0],
            self.counts.n_kf.V_plus[0],
        ).sum()
        assert numpy.isfinite(term)
        LL.append(term)

        # alpha KL
        term = - \
            self.q_alpha.KL(
                GammaDist(self.data.options.a_alpha, self.data.options.b_alpha))
        assert numpy.isfinite(term)
        LL.append(term)

        # beta KL
        term = - \
            self.q_beta.KL(GammaDist(self.data.options.a_beta,
                                     self.data.options.b_beta))
        assert numpy.isfinite(term)
        LL.append(term)

        # gamma KL
        term = - \
            self.q_gamma.KL(
                GammaDist(self.data.options.a_gamma, self.data.options.b_gamma))
        assert numpy.isfinite(term)
        LL.append(term)

        # lambda KL
        term = - \
            self.q_lambda.KL(
                GammaDist(self.data.options.a_lambda, self.data.options.b_lambda))
        assert numpy.isfinite(term)
        LL.append(term)

        # pi_bar KL
        term = -(self.q_pi_bar.KL(BetaDist(1., self.q_gamma.E))).sum()
        assert numpy.isfinite(term)
        LL.append(term)

        # tau KL
        term = -self.q_tau.KL(DirichletDist(self.data.options.a_tau))
        assert numpy.isfinite(term)
        LL.append(term)

        # omega KL
        term = -self.q_omega.KL(DirichletDist(self.data.options.a_omega))
        assert numpy.isfinite(term)
        LL.append(term)

        return LL

    def log_likelihood(self):
        "@return: The variational bound on the log likelihood."
        return sum(self._log_likelihood())

    def exp_theta(self):
        "@return: The expected value of theta, the documents' distributions over topics, given dpm.counts.E_n_gk."
        result = (self.counts.n_gk.E.T / self.counts.n_gk.E.sum(axis=1)).T
        array_replace(result, numpy.isnan(result), 0.)
        return result

    def exp_Theta(self):
        "@return: The average value of theta across all documents."
        return self.counts.n_gk.E.sum(axis=0) / self.counts.n_gk.E.sum()

    def exp_phi(self):
        "@return: The expected value of phi, the topics' distributions over words, given dpm.counts.E_n_kf."
        return (self.counts.n_kf.E.T / self.counts.n_kf.E.sum(axis=1)).T

    def exp_Phi(self):
        "@return: The empirical distribution of words over all documents and topics."
        return self.counts.n_kf.E.sum(axis=0) / self.counts.n_kf.E.sum()
