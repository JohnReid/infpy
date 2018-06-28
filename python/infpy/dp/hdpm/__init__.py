#
# Copyright John Reid 2008, 2009, 2010
#


"""
Implementation of hierarchical Dirichlet processes as detailed in
http://www.gatsby.ucl.ac.uk/~ywteh/research/inference/nips2007b.pdf
"""

from math import *
from math import _greater_than, _permutation_by_sort, _permute, _approx_f_n, _gamma_KL, _dirichlet_KL, _beta_KL, _safe_x_log_x
from utils import *
from summarise import *
import types


class ApproxCounts(object):
    "Holds approximate counts that are recalculated after q_z has been updated."

    def _check_shape(self, hdpm):
        assert self.E_n_dk.shape == (hdpm.D, hdpm.K)
        assert self.V_n_dk.shape == (hdpm.D, hdpm.K)
        assert self.Z_n_dk.shape == (hdpm.D, hdpm.K)
        assert self.P_plus_n_dk.shape == (hdpm.D, hdpm.K)
        assert self.E_plus_n_dk.shape == (hdpm.D, hdpm.K)
        assert self.V_plus_n_dk.shape == (hdpm.D, hdpm.K)

        assert self.E_n_kw.shape == (hdpm.K, hdpm.W)
        assert self.V_n_kw.shape == (hdpm.K, hdpm.W)
        assert self.Z_n_kw.shape == (hdpm.K, hdpm.W)
        assert self.P_plus_n_kw.shape == (hdpm.K, hdpm.W)
        assert self.E_plus_n_kw.shape == (hdpm.K, hdpm.W)
        assert self.V_plus_n_kw.shape == (hdpm.K, hdpm.W)

        assert self.E_n_k.shape == (hdpm.K,)
        assert self.V_n_k.shape == (hdpm.K,)
        assert self.Z_n_k.shape == (hdpm.K,)
        assert self.P_plus_n_k.shape == (hdpm.K,)
        assert self.E_plus_n_k.shape == (hdpm.K,)
        assert self.V_plus_n_k.shape == (hdpm.K,)

    def __init__(self, hdpm):
        hdpm._check_qz()

        self.E_n_kw = numpy.zeros((hdpm.K, hdpm.W))
        self.V_n_kw = numpy.zeros((hdpm.K, hdpm.W))
        self.Z_n_kw = numpy.zeros((hdpm.K, hdpm.W))
        self.E_n_dk = numpy.empty((hdpm.D, hdpm.K))
        self.V_n_dk = numpy.empty((hdpm.D, hdpm.K))
        self.Z_n_dk = numpy.empty((hdpm.D, hdpm.K))

        # for each document
        for d, (q_zd, document) in enumerate(zip(hdpm.q_z, hdpm.documents)):
            q_zd_sum = q_zd.sum(axis=0)
            not_q_zd = 1. - q_zd
            log_not_q_zd = numpy.log(not_q_zd)
            log_not_q_zd_sum = log_not_q_zd.sum(axis=0)
            V = q_zd * not_q_zd
            V_sum = V.sum(axis=0)
            #import IPython; IPython.Debugger.Pdb().set_trace()
            self.E_n_dk[d] = q_zd_sum
            self.V_n_dk[d] = V_sum
            self.Z_n_dk[d] = log_not_q_zd_sum
            assert numpy.isfinite(self.E_n_dk[d]).all()
            assert numpy.isfinite(self.V_n_dk[d]).all()
            # assert numpy.isfinite(self.Z_n_dk[d]).all() # added for without memoisation test not sure we need this
            # for each word in the document
            for i, w in enumerate(document):
                self.E_n_kw[:, w] += q_zd[i]
                self.V_n_kw[:, w] += V[i]
                self.Z_n_kw[:, w] += log_not_q_zd[i]
        assert numpy.isfinite(self.E_n_kw).all()
        assert numpy.isfinite(self.V_n_kw).all()

        self.E_n_k = self.E_n_dk.sum(axis=0)
        self.V_n_k = self.V_n_dk.sum(axis=0)
        self.Z_n_k = self.Z_n_dk.sum(axis=0)
        assert numpy.isfinite(self.E_n_k).all()
        assert numpy.isfinite(self.V_n_k).all()

        assert numpy.allclose(self.E_n_k, self.E_n_kw.sum(axis=1))
        assert numpy.allclose(self.V_n_k, self.V_n_kw.sum(axis=1))
        assert numpy.allclose(self.Z_n_k, self.Z_n_kw.sum(axis=1))

        self.P_plus_n_dk, self.E_plus_n_dk, self.V_plus_n_dk = calculate_P_plus(
            self.E_n_dk, self.V_n_dk, self.Z_n_dk)
        self.P_plus_n_kw, self.E_plus_n_kw, self.V_plus_n_kw = calculate_P_plus(
            self.E_n_kw, self.V_n_kw, self.Z_n_kw)
        self.P_plus_n_k, self.E_plus_n_k, self.V_plus_n_k  = calculate_P_plus(
            self.E_n_k, self.V_n_k, self.Z_n_k)

    def __str__(self):
        return 'ApproxCounts object'

    def permute(self, permutation):
        "Permute the counts based on a permutation of the topics."
        self.E_n_dk = _permute(self.E_n_dk, permutation, axis=1)
        self.E_n_k  = _permute(self.E_n_k, permutation, axis=0)
        self.E_n_kw = _permute(self.E_n_kw, permutation, axis=0)
        self.E_plus_n_dk = _permute(self.E_plus_n_dk, permutation, axis=1)
        self.E_plus_n_k  = _permute(self.E_plus_n_k, permutation, axis=0)
        self.E_plus_n_kw = _permute(self.E_plus_n_kw, permutation, axis=0)
        self.V_n_dk = _permute(self.V_n_dk, permutation, axis=1)
        self.V_n_k  = _permute(self.V_n_k, permutation, axis=0)
        self.V_n_kw = _permute(self.V_n_kw, permutation, axis=0)
        self.V_plus_n_dk = _permute(self.V_plus_n_dk, permutation, axis=1)
        self.V_plus_n_k  = _permute(self.V_plus_n_k, permutation, axis=0)
        self.V_plus_n_kw = _permute(self.V_plus_n_kw, permutation, axis=0)
        self.Z_n_dk = _permute(self.Z_n_dk, permutation, axis=1)
        self.Z_n_k  = _permute(self.Z_n_k, permutation, axis=0)
        self.Z_n_kw = _permute(self.Z_n_kw, permutation, axis=0)
        self.P_plus_n_dk = _permute(self.P_plus_n_dk, permutation, axis=1)
        self.P_plus_n_k  = _permute(self.P_plus_n_k, permutation, axis=0)
        self.P_plus_n_kw = _permute(self.P_plus_n_kw, permutation, axis=0)


def _examine_document_for_duplicate_words(document):
    """
    Look in the document for duplicated words and build a list that maps duplicates.

    This list is used to optimise the HDPM._calculate_z() function.
    """
    first_occurrence = dict()
    duplicates = list()

    # for each word
    for i, w in enumerate(document):
        if w in first_occurrence:
            # we have already seen this word in this document.
            duplicates.append(first_occurrence[w])
        else:
            first_occurrence[w] = i
            duplicates.append(None)

    return duplicates


class HDPM(object):
    """
    Implementation of hierarchical Dirichlet processes as detailed in
    http://www.gatsby.ucl.ac.uk/~ywteh/research/inference/nips2007b.pdf
    """

    INIT_METHOD_SIMPLE = 1
    INIT_METHOD_SAMPLE = 2

    def _memoize_calculations(self):
        """
        Memoize those calculations we might otherwise make over and over.
        """
        self._calculate_E_s_dk      = memoize(self._calculate_E_s_dk)
        self._calculate_E_t_kw      = memoize(self._calculate_E_t_kw)
        self._calculate_E_log_xi    = memoize(self._calculate_E_log_xi)
        self._calculate_E_log_eta   = memoize(self._calculate_E_log_eta)
        self._calculate_G_pi        = memoize(self._calculate_G_pi)
        self._calculate_counts      = memoize(self._calculate_counts)

    def __getstate__(self):
        state = {}
        for attr, value in self.__dict__.iteritems():
            if not isinstance(value, Memoize):
                state[attr] = value
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._memoize_calculations()

    def __init__(
        self,
        documents,
        W,
        K,
        a_alpha=4.,
        b_alpha=4.,
        a_beta=4.,
        b_beta=4.,
        a_gamma=7.5,
        b_gamma=7.5,
        a_tau=None,
        init_method=INIT_METHOD_SAMPLE,
        init_method_args=None
    ):
        """
        Initialise the hierarchical Dirichlet process mixture.

        The values are initialised as detailed in the paper (w/out the elaborate scheme, we just use
        the mid-point of the ranges they specify)
        """

        self._memoize_calculations()

        if 0 == len(documents):
            raise ValueError('No data: expecting at least one document.')

        self.W = W
        "Size of the vocabulary"

        word_dist, N = calculate_corpus_wide_word_distribution(documents, W)
        self.N = N
        "Number of words in the documents."

        # set default a_tau if necessary
        if None == a_tau:
            a_tau = 1.

        self.a_alpha = a_alpha
        "Parameter for prior on alpha"

        self.b_alpha = b_alpha
        "Parameter for prior on alpha"

        self.a_beta = a_beta
        "Parameter for prior on beta"

        self.b_beta = b_beta
        "Parameter for prior on beta"

        self.a_gamma = a_gamma
        "Parameter for prior on gamma"

        self.b_gamma = b_gamma
        "Parameter for prior on gamma"

        self.a_tau = a_tau
        "Parameter for prior on tau"

        self.documents = documents
        "Documents."
        for d, document in enumerate(self.documents):
            if not len(document):
                raise ValueError('Document %d is empty' % d)

        self._duplicates = [_examine_document_for_duplicate_words(
            document) for document in self.documents]
        "Maps of duplicates used to optimise _calculate_z()."

        self.K = K
        "Max number of topics"

        self.D = len(self.documents)
        "Number of documents"

        self.n_d = numpy.array([len(document) for document in self.documents])
        "Number of words in each document"

        # initialise the variational distributions
        if None == init_method_args:
            init_method_args = tuple()
        if HDPM.INIT_METHOD_SAMPLE == init_method:
            self.initialise_hdpm_variational_by_sampling(*init_method_args)
        elif HDPM.INIT_METHOD_SIMPLE == init_method:
            self.initialise_hdpm_variational_simply(*init_method_args)
        else:
            raise ValueError(
                'Unrecognized variational distribution initialisation method.')

        # check arrays have correct dimensions
        self._check_shapes()

    def initialise_hdpm_variational_simply(self, q_z_dirichlet_param=1.):
        "Old straightforward way to initialise the variational distributions."

        self.q_alpha = GammaDist(self.a_alpha, self.b_alpha)
        "Variational distribution over alpha"

        self.q_beta = GammaDist(self.a_beta, self.b_beta)
        "Variational distribution over beta"

        self.q_gamma = GammaDist(self.a_gamma, self.b_gamma)
        "Variational distribution over gamma"

        self.q_pi_bar = BetaDist(numpy.ones(
            self.K), self.a_gamma / self.b_gamma * numpy.ones(self.K))
        "Variational distribution over pi_bar"

        self.q_tau = DirichletDist(numpy.ones(self.W))
        "Variational distribution over tau"

        self.q_z = [
            numpy.random.dirichlet(numpy.ones(
                self.K) * q_z_dirichlet_param, size=len(document))
            for document
            in self.documents
        ]
        "Variational distribution over z_id, indexed by d."
        self._normalise_z()

    def initialise_hdpm_variational_by_sampling(self):
        """
        Initialise the variational distributions of a HDPM by sampling the generative model (ignoring the data).
        """

        self.q_gamma = GammaDist(self.a_gamma, self.b_gamma)
        "Variational distribution over gamma"
        gamma = self.q_gamma.sample()

        self.q_pi_bar = BetaDist(numpy.ones(
            self.K), gamma * numpy.ones(self.K))
        "Variational distribution over pi_bar"
        pi_bar = self.q_pi_bar.sample()
        pi = numpy.empty_like(pi_bar)
        for k in xrange(self.K - 1):
            pi[k] = pi_bar[k] * (1. - pi_bar[:k]).prod()
        pi[-1] = 1. - pi[:-1].sum()
        if pi[-1] < 0.:  # adjust for numerical errors
            pi[-1] = 0.

        self.q_alpha = GammaDist(self.a_alpha, self.b_alpha)
        "Variational distribution over alpha"
        alpha = self.q_alpha.sample()

        alpha_pi = alpha * pi
        for document in self.documents:
            numpy.random.dirichlet(alpha_pi, size=len(document))
        self.q_z = [
            numpy.random.dirichlet(alpha_pi, size=len(document))
            for document in self.documents
        ]
        "Variational distribution over z_id, indexed by d."
        self._normalise_z()

        self.q_beta = GammaDist(self.a_beta, self.b_beta)
        "Variational distribution over beta"

        self.q_tau = DirichletDist(numpy.ones(self.W))
        "Variational distribution over tau"

        return gamma, pi_bar, pi, alpha

    def _calculate_E_s_dk(self):
        G_alpha_pi = self.q_alpha.G * self.G_pi
        term = G_alpha_pi + self.counts.E_plus_n_dk
        E_s_dk = G_alpha_pi * self.counts.P_plus_n_dk * (
            digamma(term)
            - digamma(G_alpha_pi)
            + .5 * self.counts.V_plus_n_dk * polygamma(2, term)
        )
        # adjust for divide by zero errors when P_plus_n_dk==0.
        array_replace(E_s_dk, self.counts.P_plus_n_dk == 0, 0.)
        assert numpy.isfinite(E_s_dk).all()
        return E_s_dk

    def _calculate_E_t_kw(self):
        G_beta_tau = self.q_beta.G * self.q_tau.G
        term = G_beta_tau + self.counts.E_plus_n_kw
        E_t_kw = G_beta_tau * self.counts.P_plus_n_kw * (
            digamma(term)
            - digamma(G_beta_tau)
            + .5 * self.counts.V_plus_n_kw * polygamma(2, term)
        )
        # adjust for divide by zero errors when P_plus_n_dk==0.
        array_replace(E_t_kw, self.counts.P_plus_n_kw == 0., 0.)
        assert numpy.isfinite(E_t_kw).all()
        assert (E_t_kw >= 0.).all()
        return E_t_kw

    def _calculate_E_log_xi(self):
        term = self.q_beta.E + self.counts.E_plus_n_k
        E_log_xi = self.counts.P_plus_n_k * (
            digamma(self.q_beta.E)
            - digamma(term)
            - .5 * self.counts.V_plus_n_k * polygamma(2, term)
        )
        # adjust for divide by zero errors when P_plus_n_dk==0.
        array_replace(E_log_xi, self.counts.P_plus_n_k == 0., 0.)
        assert numpy.isfinite(E_log_xi).all()
        return E_log_xi

    def _calculate_E_log_eta(self):
        return digamma(self.q_alpha.E) - digamma(self.q_alpha.E + self.n_d)

    def _calculate_G_pi(self):
        accum = numpy.empty((self.K,))
        accum[0] = 1.
        for k in xrange(1, self.K):
            accum[k] = accum[k - 1] * self.q_1_minus_pi_bar.G[k - 1]
        G_pi = self.q_pi_bar.G * accum
        assert numpy.isfinite(G_pi).all()
        assert G_pi.sum() > 0.
        self._calculate_E_s_dk.clear_cached_value()
        return G_pi

    def _calculate_counts(self):
        self._check_qz()
        counts = ApproxCounts(self)
        self._calculate_E_s_dk.clear_cached_value()
        self._calculate_E_t_kw.clear_cached_value()
        self._calculate_E_log_xi.clear_cached_value()
        return counts

    def _cachable_fns(self):
        return (
            self._calculate_E_s_dk,
            self._calculate_E_t_kw,
            self._calculate_E_log_xi,
            self._calculate_E_log_eta,
            self._calculate_G_pi,
            self._calculate_counts,
        )

    def _get_E_s_dk(self):
        return self._calculate_E_s_dk()

    def _get_E_t_kw(self):
        return self._calculate_E_t_kw()

    def _get_E_log_xi(self):
        return self._calculate_E_log_xi()

    def _get_E_log_eta(self):
        return self._calculate_E_log_eta()

    def _get_G_pi(self):
        return self._calculate_G_pi()

    def _get_counts(self):
        return self._calculate_counts()

    def _set_q_alpha(self, q_alpha):
        self._q_alpha = q_alpha
        self._calculate_E_s_dk.clear_cached_value()
        self._calculate_E_log_eta.clear_cached_value()

    def _set_q_beta(self, q_beta):
        self._q_beta = q_beta
        self._calculate_E_t_kw.clear_cached_value()
        self._calculate_E_log_xi.clear_cached_value()

    def _set_q_gamma(self, q_gamma):
        self._q_gamma = q_gamma

    def _set_q_pi_bar(self, q_pi_bar):
        self._q_pi_bar = q_pi_bar
        self._q_1_minus_pi_bar = BetaDist(self.q_pi_bar.b, self.q_pi_bar.a)
        self._calculate_G_pi.clear_cached_value()
        self._calculate_E_s_dk.clear_cached_value()

    def _set_q_tau(self, q_tau):
        self._q_tau = q_tau
        self._calculate_E_t_kw.clear_cached_value()

    def _update_alpha(self):
        self.q_alpha = GammaDist(
            self.a_alpha + self.E_s_dk.sum(),
            self.b_alpha - self.E_log_eta.sum()
        )

    def _update_beta(self):
        self.q_beta = GammaDist(
            self.a_beta + self.E_t_kw.sum(),
            self.b_beta - self.E_log_xi.sum()
        )

    def _update_gamma(self):
        self.q_gamma = GammaDist(
            self.a_gamma + self.K,
            self.b_gamma - numpy.log(self.q_1_minus_pi_bar.G).sum()
        )

    def _update_pi_bar(self):
        E_s_k = self.E_s_dk.sum(axis=0)
        assert E_s_k.shape == (self.K,)
        E_s_greater_k = _greater_than(E_s_k)
        self.q_pi_bar = BetaDist(E_s_k + 1., self.q_gamma.E + E_s_greater_k)

    def _update_tau(self):
        self.q_tau = DirichletDist(self.a_tau + self.E_t_kw.sum(axis=0))

    def _calculate_z(self):
        """
        Update our estimates of which topics each word belongs to.

        Profiling has shown that HDPM learning typically spends over half its time in the function.
        """
        counts = self.counts
        G_pi = self.G_pi
        q_alpha = self.q_alpha
        q_beta = self.q_beta
        q_tau = self.q_tau

        assert numpy.isfinite(counts.V_n_k).all()

        # for each document
        for d, (document, q_zd, duplicates) in enumerate(zip(self.documents, self.q_z, self._duplicates)):

            assert numpy.isfinite(counts.V_n_dk[d]).all()

            # calculate some things that don't depend on the words...
            dk_term_half = q_alpha.G * G_pi + counts.E_n_dk[d]
            V_n_dk = counts.V_n_dk[d]
            k_term_half = q_beta.E + counts.E_n_k

            # for each word in the document
            for w, q_zdi, dupe in zip(document, q_zd, duplicates):

                # have we already calculated q_z for the same word in this document?
                if None != dupe:

                    # yes so just copy the result
                    q_zdi[:] = q_zd[dupe]

                else:

                    # no do the calculation
                    assert numpy.isfinite(counts.V_n_kw[:, w]).all()

                    V = q_zdi * (1. - q_zdi)
                    assert numpy.isfinite(V).all()

                    # calculate the terms in the exponent, making sure they are finite
                    dk_term = dk_term_half - q_zdi
                    dk_top = V_n_dk - V
                    dk_exp = dk_top / dk_term**2
                    array_replace(dk_exp, 0. == dk_top, 0.)
                    assert numpy.isfinite(dk_term).all()

                    kw_term = q_beta.G * \
                        q_tau.G[w] + counts.E_n_kw[:, w] - q_zdi
                    kw_top = counts.V_n_kw[:, w] - V
                    kw_exp = kw_top / kw_term**2
                    array_replace(kw_exp, 0. == kw_top, 0.)
                    assert numpy.isfinite(kw_term).all()

                    k_term = k_term_half - q_zdi
                    k_top = counts.V_n_k - V
                    k_exp = k_top / k_term**2
                    array_replace(k_exp, 0. == k_top, 0.)
                    assert numpy.isfinite(k_term).all()

                    # the total exponent
                    exponent = (- dk_exp - kw_exp + k_exp) / 2.
                    #assert numpy.isfinite(exponent).all()

                    q_zdi[:] = dk_term * kw_term / k_term * numpy.exp(exponent)
                    # adjust for very large negative exponents. This makes q_zdi 0.
                    array_replace(q_zdi[:], numpy.isneginf(exponent), 0.)
                    assert numpy.isfinite(q_zdi).all()
                    assert (q_zdi >= 0.).all()
                    assert q_zdi.sum() > 0.

        self._normalise_z()

    def _normalise_z(self):
        "Normalises q(z) so that each word's probabilities sum to 1."
        for d, q_zd in enumerate(self.q_z):
            for k, q_zdk in enumerate(q_zd):
                total = q_zdk.sum()
                assert 0.0 != total
                q_zdk /= total
        self._check_qz()

    def _check_qz(self):
        """
        Check that the q_z are in the right range [0,1].
        """
        for q_zd in self.q_z:
            assert numpy.isfinite(q_zd).all()
            assert (0. <= q_zd).all()
            assert (q_zd <= 1.).all()

    def _update_z(self):
        self._calculate_z()
        self._calculate_counts.clear_cached_value()

    def _reorder_topic_labels(self):
        "Reorder the topic labels so that the largest topic is always first."

        # calculate the permutation required to put the topics in order of size.
        permutation = _permutation_by_sort(self.counts.E_n_k)

        # don't bother re-ordering anything if permutation is the identity
        if (numpy.arange(self.K) != permutation).any():

            #start_LL = numpy.array(self._log_likelihood())

            for d, q_zd in enumerate(self.q_z):
                self.q_z[d] = _permute(q_zd, permutation, axis=1)

            a_permuted = _permute(self.q_pi_bar.a, permutation, axis=0)
            b_permuted = _permute(self.q_pi_bar.b, permutation, axis=0)
            self.q_pi_bar.a = a_permuted
            self.q_pi_bar.b = b_permuted
            self.q_1_minus_pi_bar.a = b_permuted
            self.q_1_minus_pi_bar.b = a_permuted

            # clear the cached values (they all depend on the topic ordering)
            # could also permute them rather than recalculating - that might
            # be more efficient.
            self._calculate_E_s_dk.cached_value = _permute(
                self.E_s_dk, permutation, axis=1)
            self._calculate_E_t_kw.cached_value = _permute(
                self.E_t_kw, permutation, axis=0)
            self._calculate_E_log_xi.cached_value = _permute(
                self.E_log_xi, permutation, axis=0)
            self._calculate_G_pi.cached_value = _permute(
                self.G_pi, permutation, axis=0)
            self._calculate_E_log_eta.clear_cached_value()
            self.counts.permute(permutation)

            # check topics are sorted in decreasing order of size
            assert (self.counts.E_n_k[:-1] >= self.counts.E_n_k[1:]).all()

            #end_LL = numpy.array(self._log_likelihood())
            #indices = [0, 1, 2, 3, 4, 5, 6, 8]
            #assert numpy.allclose(start_LL[indices], end_LL[indices])
            #logging.info('7: %f', end_LL[7] - start_LL[7])

            # this re-ordering seems to affect the LL badly sometimes, updating pi bar afterwards seems to
            # help. I'm not sure if the decrease in the LL is due to the approximation or a bug I can't find
            # self._update_pi_bar()

    def update(self):
        "Update the model. That is do one iteration of learning."
        for update_fn in HDPM.update_fns:
            update_fn(self)

    update_fns = (
        _update_gamma,
        _update_pi_bar,
        _update_alpha,
        _update_z,
        _reorder_topic_labels,
        _update_beta,
        _update_tau,
    )
    "The update functions for the HDPM. You can change the order in which the variational distributions are updated by reordering this list."

    E_s_dk      = property(_get_E_s_dk, None, None,
                           "Expected s for given d and k.")
    E_t_kw      = property(_get_E_t_kw, None, None,
                           "Expected t for given k and w.")
    E_log_xi    = property(_get_E_log_xi, None, None, "Expected log x_i.")
    E_log_eta   = property(_get_E_log_eta, None, None, "Expected log eta.")
    G_pi        = property(_get_G_pi, None, None,
                           "Geometrical expectation of pi.")
    counts      = property(_get_counts, None, None, "Approximate counts.")

    q_alpha = property(
        lambda self: self._q_alpha,
        _set_q_alpha,
        None,
        "Variational distribution over alpha."
    )

    q_beta = property(
        lambda self: self._q_beta,
        _set_q_beta,
        None,
        "Variational distribution over beta."
    )

    q_gamma = property(
        lambda self: self._q_gamma,
        _set_q_gamma,
        None,
        "Variational distribution over gamma."
    )

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

    q_tau = property(
        lambda self: self._q_tau,
        _set_q_tau,
        None,
        "Variational distribution over tau."
    )

    def _check_shapes(self):
        """
        Check that the parameters are correctly dimensioned
        """
        assert len(self.documents) == self.D
        assert self.n_d.shape == (self.D,)

        assert self.q_tau.G.shape == (self.W,)
        assert self.G_pi.shape == (self.K,)

        assert len(self.q_z) == self.D
        for q_zd, document in zip(self.q_z, self.documents):
            assert q_zd.shape == (len(document), self.K)

        assert self.E_s_dk.shape == (self.D, self.K)
        assert self.E_t_kw.shape == (self.K, self.W)
        assert self.E_log_eta.shape == (self.D,)
        assert self.E_log_xi.shape == (self.K,)

        self.counts._check_shape(self)

    def _log_likelihood(self):
        """
        Returns the variational bound on the log likelihood.
        """
        LL = []  # build a list of all the different terms in the LL

        # n_d term
        term = (gammaln(self.q_alpha.E) -
                gammaln(self.q_alpha.E + self.n_d)).sum()
        assert numpy.isfinite(term)
        LL.append(term)

        # n_dk term
        G_alpha_pi = self.q_alpha.G * self.G_pi

        def f(n): return gammaln(G_alpha_pi + n) - gammaln(G_alpha_pi)

        def f2(n): return polygamma(1, (G_alpha_pi + n))
        term = _approx_f_n(f, f2, self.counts.P_plus_n_dk,
                           self.counts.E_plus_n_dk, self.counts.V_plus_n_dk).sum()
        assert numpy.isfinite(term)
        LL.append(term)

        # n_k term
        def f(n): return gammaln(self.q_beta.G + n) - gammaln(self.q_beta.G)

        def f2(n): return polygamma(1, (self.q_beta.G + n))
        term = - _approx_f_n(f, f2, self.counts.P_plus_n_k,
                             self.counts.E_plus_n_k, self.counts.V_plus_n_k).sum()
        assert numpy.isfinite(term)
        LL.append(term)

        # n_kw term
        G_beta_tau = self.q_beta.G * self.q_tau.G

        def f(n): return gammaln(G_beta_tau + n) - gammaln(G_beta_tau)

        def f2(n): return polygamma(1, (G_beta_tau + n))
        term = _approx_f_n(f, f2, self.counts.P_plus_n_kw,
                           self.counts.E_plus_n_kw, self.counts.V_plus_n_kw).sum()
        assert numpy.isfinite(term)
        LL.append(term)

        # KL(q(alpha)||p(alpha)) term
        a_bar = self.a_alpha + self.E_s_dk.sum()
        b_bar = self.b_alpha - self.E_log_eta.sum()
        term = - _gamma_KL(a_bar, b_bar, self.a_alpha, self.b_alpha)
        assert numpy.isfinite(term)
        LL.append(term)

        # KL(q(z)||p(z)) term
        term = sum(- _safe_x_log_x(q_zd).sum() for q_zd in self.q_z)
        LL.append(term)
        assert numpy.isfinite(term)

        # KL(q(beta)||p(beta)) term
        a_bar = self.a_beta + self.E_t_kw.sum()
        b_bar = self.b_beta - self.E_log_xi.sum()
        term = - _gamma_KL(a_bar, b_bar, self.a_beta, self.b_beta)
        assert numpy.isfinite(term)
        LL.append(term)

        # KL(q(pi)||p(pi)) term
        E_s_k = self.E_s_dk.sum(axis=0)
        assert E_s_k.shape == (self.K,)
        E_s_greater_k = _greater_than(E_s_k)
#        term = - (
#            gammaln(1. + self.q_gamma.E + E_s_k + E_s_greater_k)
#            - numpy.log(self.q_gamma.E)
#            - gammaln(1. + E_s_k)
#            - gammaln(self.q_gamma.E + E_s_greater_k)
#            + E_s_k * numpy.log(self.q_pi_bar.G)
#            + E_s_greater_k * numpy.log(self.q_1_minus_pi_bar.G)
#        ).sum()
        term = - (
            (
                gammaln(1. + self.q_gamma.E + E_s_k + E_s_greater_k)
                - numpy.log(self.q_gamma.E)
                - gammaln(1. + E_s_k)
                - gammaln(self.q_gamma.E + E_s_greater_k)
            )
            * self.q_pi_bar.G ** E_s_k
            * self.q_1_minus_pi_bar.G ** E_s_greater_k
        ).sum()
        #assert numpy.allclose(term, _beta_KL(self.q_pi_bar.a, self.q_pi_bar.b, 1., self.q_gamma.E))
        #term = _beta_KL(self.q_pi_bar.a, self.q_pi_bar.b, 1., self.q_gamma.G).sum()
        assert numpy.isfinite(term)
        LL.append(term)

        # KL(q(tau)||p(tau)) term
        a_bar = self.q_tau.a + self.E_t_kw.sum(axis=0)
        term = - _dirichlet_KL(a_bar, self.q_tau.a)
        assert numpy.isfinite(term)
        LL.append(term)

        return LL

    def log_likelihood(self):
        "@return: The variational bound on the log likelihood."
        return sum(self._log_likelihood())

    def entropy(self):
        """
        Returns the entropy of the variational distribution as a tuple
        """
        return (
            -sum(_safe_x_log_x(q_zd).sum() for q_zd in self.q_z),
            self.q_alpha.H,
            self.q_beta.H,
            self.q_gamma.H,
            self.q_pi_bar.H.sum(),
            self.q_tau.H,
        )

    def exp_theta(self):
        "@return: The expected value of theta, the documents' distributions over topics, given dpm.counts.E_n_dk."
        result = (self.counts.E_n_dk.T / self.counts.E_n_dk.sum(axis=1)).T
        array_replace(result, numpy.isnan(result), 0.)
        return result

    def exp_Theta(self):
        "@return: The average value of theta across all documents."
        return self.counts.E_n_dk.sum(axis=0) / self.counts.E_n_dk.sum()

    def exp_phi(self):
        "@return: The expected value of phi, the topics' distributions over words, given dpm.counts.E_n_kw."
        return (self.counts.E_n_kw.T / self.counts.E_n_kw.sum(axis=1)).T

    def exp_Phi(self):
        "@return: The empirical distribution of words over all documents and topics."
        return self.counts.E_n_kw.sum(axis=0) / self.counts.E_n_kw.sum()


def test_permutation():
    a = numpy.array((3, 2, 1, 6, 5, 4, 10, 0))
    p = _permutation_by_sort(a)
    a_permuted = _permute(a, p)  # b should be in decreasing order
    b = numpy.array([a + 100 * i for i in xrange(4)])
    # b's second axis should be in decreasing order
    b_permuted = _permute(b, p, axis=1)
    return p, a, a_permuted, b, b_permuted


def test_dpm():
    numpy.random.seed(1)
    W = 4
    K = 8
    documents = [
        numpy.array([1]),
        numpy.array([1, 1, 2, 3, 2, 3, 2, 3]),
        numpy.array([1, 1, 2, 3, 2, 3, 2, 3]),
        numpy.array([2, 0, 0, 2, 2, 0, 0, 0, 2, 2, 2, 2]),
        numpy.array([1, 1, 2, 3, 2, 3, 2, 3, 2, 2, 0, 0, 0, 2]),
        numpy.array([1, 1, 2, 3, 2, 3, 2, 3, 2, 2, 0, 0, 0, 2]),
    ]
    dpm = HDPM(documents=documents, K=K, W=W)
    for _i in xrange(60):
        # print dpm.entropy()
        dpm.update()
        print dpm.log_likelihood()
    print 'Expected # words in each topic'
    print dpm.counts.E_n_dk.sum(axis=0)
    print 'Topic per document distributions'
    print (dpm.counts.E_n_dk.T / dpm.counts.E_n_dk.sum(axis=1)).T
    print 'Topic distributions'
    print (dpm.counts.E_n_kw.T / dpm.counts.E_n_kw.sum(axis=1)).T


def sample_chinese_restaurant(alpha, N):
    """
    Sample N times from a Chinese restaurant process with parameter alpha.

    @return: list of partition sizes
    """
    import numpy.random as R
    partition_sizes = [float(alpha)]
    for n in xrange(N):
                # choose which partition it came from
        multi_sample = R.multinomial(
            1, numpy.array(partition_sizes) / (alpha + n))
        partition = numpy.where(multi_sample)[0][0]

        # was it a new partition?
        new_partition = partition == 0
        if new_partition:
            partition_sizes.append(1)
        else:
            partition_sizes[partition] += 1

    # take alpha off the beginning and return
    del partition_sizes[0]
    return partition_sizes


if '__main__' == __name__:
    p, a, a_permuted, b, b_permuted = test_permutation()
    test_dpm()
