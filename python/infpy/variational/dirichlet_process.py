#
# Copyright John Reid 2006
#

"""See Blei, Jordan - Variational Methods for Dirichlet Processes"""

import numpy
import scipy.special
import math
import infpy


class ExpFamilyDP(object):
    """
    Implements a variational algorithm for inference in a dirichlet process
    mixture model of exponential family distributions.
    """

    def __init__(
            self,
            X,
            K,
            alpha,
            family,
            lambda_
    ):
        """Initialise the algorithm

        @param X: The data (in canonical form, i.e. not sufficient statistics)
        @param K: Variational truncation parameter
        @param alpha: Parameter for dirichlet process
        @param family: The exponential family
        @param lambda_: Conjugate prior for exp family parameters
        """

        self.gamma = None
        "The variational parameters for the V's (array of shape (K,2))."

        self.tau = None
        "The variational parameters for the eta's (array of length K)."

        self.tau_0 = None
        "The pseudo count variational parameters for the eta's (array of length K)."

        self.phi = None
        "The variational parameters for the Z's (array of shape (N,2))."

        self._do_ll_checks = True
        "Check the log likelihood increases."

        self.family = family
        "Exponential family."

        self.K = K
        "Variational # mixtures truncation parameter."

        self.N = len(X)
        "# of data."

        self.alpha = alpha
        "Dirichlet process parameter."

        self.data = [
            self.family.T(x)
            for x
            in X
        ]
        "The data as sufficient statistics."

        self.lambda_ = lambda_
        "Conjugate prior parameter."

        self.initialise_variational_parameters()

    def initialise_variational_parameters(self):
        """Set the variational parameters to some starting point."""

        # gamma will be updated first so set it to anything...
        self.gamma = numpy.ones((self.K - 1, 2), dtype=numpy.float64)

        self.phi = numpy.random.uniform(size=(self.N, self.K))
        self._normalise_phi()
        self.tau = [
            (
                self.lambda_[0],
                self.lambda_[1]
            )
            for i in range(self.K)
        ]
        # assign each data point to the mixtures according to random phi
        for n in range(self.N):
            if len(self.data[n]) != len(self.lambda_[1]):
                raise RuntimeError(
                    'Sufficient statistic length (%d) and prior length (%d) differ' % (
                        (len(self.data[n]), len(self.lambda_[1]))
                    )
                )
            for i in range(self.K):
                self.tau[i] = (
                    self.tau[i][0] + self.phi[n, i],
                    self.tau[i][1] + self.phi[n, i] * self.data[n]
                )

    def update(self):
        """Perform one update step."""
        self.update_gamma()
        self.update_tau()
        self.update_phi()

    def _get_LL_check(self):
        """
        If log likelihood checking is switched
        on this returns the LL otherwise None
        """
        if hasattr(self, '_do_ll_checks') and self._do_ll_checks:
            return self.log_likelihood()
        else:
            return None

    def _check_LL(self, last_LL):
        """
        If last_LL != None then checks that it is greater than current LL
        """
        if None != last_LL:
            LL = self.log_likelihood()
            if last_LL > LL and not infpy.check_is_close(last_LL, LL):
                raise RuntimeError(
                    'LL has decreased by %g. %g -> %g'
                    % (last_LL - LL, last_LL, LL)
                )

    def update_gamma(self):
        """Update gamma"""
        #LL = self._get_LL_check()
        for i in range(self.K - 1):
            self.gamma[i, 0] = 1.0 + sum(
                [
                    self.phi[n, i]
                    for n in range(self.N)
                ]
            )
            self.gamma[i, 1] = self.alpha + sum(
                [
                    sum(
                        [
                            self.phi[n, j]
                            for j in range(i + 1, self.K)
                        ]
                    )
                    for n in range(self.N)
                ]
            )
        # print 'Gamma:\n', self.gamma
        # raise
        #self._check_LL( LL )

    def update_tau(self):
        """Update tau"""
        #LL = self._get_LL_check()
        for i in range(self.K):
            # each tau is a tuple of the pseudo count and a vector
            self.tau[i] = (
                self.lambda_[0] + sum(
                    [
                        self.phi[n, i]
                        for n
                        in range(self.N)
                    ]
                ),
                self.lambda_[1] + sum(
                    [
                        self.phi[n, i] * self.data[n]
                        for n
                        in range(self.N)
                    ]
                )
            )
        # print 'Tau:\n', self.tau
        #self._check_LL( LL )

    def update_phi(self):
        """Update phi."""
        #LL = self._get_LL_check()
        digamma = scipy.special.digamma
        for i in range(self.K):
            # E(log V_i)
            if i == self.K - 1:
                E_log_V_i = 0.0  # last mixture's v is always 1
            else:
                E_log_V_i = (
                    digamma(self.gamma[i, 0])
                    - digamma(self.gamma[i, 0] + self.gamma[i, 1])
                )

            # E(a(eta_i))
            E_a = self.family.expected_a(self.tau[i])

            # Sum of E(log(1-V_j))
            E_sum_log_1_minus_V_j = sum(
                [
                    digamma(self.gamma[j, 1])
                    - digamma(self.gamma[j, 0] + self.gamma[j, 1])
                    for j
                    in range(i - 1)
                ]
            )

            E_eta = self.family.expected_eta(self.tau[i]).T
            for n in range(self.N):
                # E(eta_i).X_n
                E_eta_dot_X = (E_eta * self.data[n])[0, 0]

                # we have to watch out for underflow - so just store E in phi
                # for the moment
                self.phi[n, i] = (
                    E_log_V_i
                    + E_eta_dot_X
                    + E_a
                    + E_sum_log_1_minus_V_j
                )
        # to cater for underflow - add constant to each row to make largest 0...
        # phi is proportional to exponent of E so is invariant to this
        for n in range(self.N):
            largest = max(self.phi[n, :])
            self.phi[n, :] -= largest
        # only now take exponent
        self.phi = numpy.exp(self.phi)
        # now normalise
        self._normalise_phi()
        #self._check_LL( LL )

    def _normalise_phi(self):
        # normalise phi
        for n in range(self.N):
            s = sum(self.phi[n, :])
            if 0.0 == s:
                # print self.phi
                raise RuntimeError('sum(phi[%d,:]) is 0' % n)
            self.phi[n, :] /= s

    def log_likelihood(self):
        """The log likelihood of the data."""
        return sum(
            [
                math.log(
                    numpy.dot(
                        self._p_T_in_mixture(self.data[n]),
                        self.phi[n, :]
                    )
                )
                for n in range(self.N)
            ]
        )
        # return sum(
        #       [
        #               math.log( self.p_T( self.data[n] ) )
        #               for n in xrange( self.N )
        #       ]
        # )

    def _gamma_factor(self, i):
        """Get the factor for the i'th gamma."""
        return self.gamma[i, 0] / (self.gamma[i, 0] + self.gamma[i, 1])

    def expected_theta(self):
        """
        Expected value of theta under the variational distribution.
        """
        result = numpy.ones(self.K, dtype=numpy.float64)
        running_product = 1.0
        for i in range(self.K - 1):
            factor = self._gamma_factor(i)
            result[i] = factor * running_product
            running_product *= (1.0 - factor)
        result[-1] = running_product
        return result

    def _p_T_in_mixture(self, T):
        """Array of pdf's of T given tau, one for each mixture.

        @param T: sufficient statistic
        """
        import infpy.exp_family
        return numpy.array(
            [
                math.exp(
                    infpy.exp_family.log_p_T_given_tau(
                        self.family, T, self.tau[i])
                )
                for i in range(self.K)
            ],
            dtype=numpy.float64
        )

    def p_T(self, T):
        """The pdf of the variational density at T.

        @param T: sufficient statistic
        """
        return numpy.dot(
            self.expected_theta(),
            self._p_T_in_mixture(T)
        )

    def p_z_given_T(self, T):
        """The pdf of variational density at z given T as an array over all z's

        @param T: sufficient statistic
        """
        expected_theta = self.expected_theta()
        p_T_in_mixture = self._p_T_in_mixture(T)
        total = numpy.dot(expected_theta, p_x_in_mixture)
        return numpy.array(
            [
                expected_theta[i] * p_T_in_mixture[i] / total
                for i in range(self.K)
            ],
            dtype=numpy.float64
        )
