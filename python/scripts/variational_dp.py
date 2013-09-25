#
# Copyright John Reid 2006
#

"""
See Blei, Jordan - Variational Methods for Dirichlet Processes.
"""

import numpy, scipy.special, math

class ExpFamilyVariationalDirichletProcess( object ):
    """
    Implements a variational algorithm for inference in a dirichlet process
    mixture model of exponential family distributions

    member K:
    N: the number of data points
    phi: the variational parameters for the Z's (array of shape (N,2))
    gamma: the variational parameters for the V's (array of shape (K,2))
    tau: the variational parameters for the eta's (array of length K)
    tau_0: the pseudo count variational parameters for the eta's (array of length K)

    i indexes mixture components
    """

    def __init__(
            self,
            data,
            K,
            alpha,
            exp_family,
            lambda_1,
            lambda_0
    ):
        """
        Initialise the algorithm.
        """

        self.exp_family = exp_family
        "The exponential family."

        self.K = K
        "Variational truncation parameter."

        self.N = len( data )
        "The number of data."

        self.alpha = alpha
        "Parameter for dirichlet process."

        self.data = [
                self.exp_family.sufficient_statistics( d )
                for d
                in data
        ]
        "The data (in natural form)."

        self.lambda_1 = lambda_1
        "Conjugate prior for exp family parameters."

        self.lambda_0 = lambda_0
        "Pseudo count of exp family conjugate prior."

        self.initialise_variational_parameters()

    def initialise_variational_parameters(self):
        """Set the variational parameters to some starting point."""
        self.gamma = numpy.zeros( ( self.K, 2 ), dtype = numpy.float64 )
        self.phi = numpy.random.uniform(
                size = self.N * self.K
        ).reshape(
                ( self.N, self.K )
        )
        self._normalise_phis()

    def update( self ):
        """Perform one update step."""
        self.update_gamma()
        self.update_tau()
        self.update_phi()

    def update_gamma( self ):
        """Update gamma."""
        for i in xrange( self.K ):
            self.gamma[i,0] = 1.0 + sum(
                    [
                            self.phi[n,i]
                            for n
                            in xrange( self.N )
                    ]
            )
            self.gamma_0[i,1] = alpha + sum(
                    [
                            sum(
                                    [
                                            self.phi[n,j]
                                            for j
                                            in xrange( n + 1, self.N )
                                    ]
                            )
                            for n
                            in xrange( self.N )
                    ]
            )

    def update_tau( self ):
        """Update tau."""
        for i in xrange( self.K ):
            self.tau[i] = self.lambda_1 + sum(
                    [
                            self.phi[n,i] * self.data[n]
                            for n
                            in xrange( self.N )
                    ]
            )
            self.tau_0[i] = self.lambda_0 + sum(
                    [
                            self.phi[n,i]
                            for n
                            in xrange( self.N )
                    ]
            )

    def update_phi( self ):
        """Update phi."""
        digamma = scipy.special.digamma
        for n in xrange( self.N ):
            for i in xrange( self.K ):
                self.phi[n,i] = math.exp(

                        # E(log V_i)
                        digamma( self.gamma[i,0] )
                        - digamma( self.gamma[i,0] + self.gamma[i,1] )

                        + # E(eta_i).X_n
                        numpy.dot(
                                self.exp_family.expectation_of_eta( self.tau[i], self.tau_0[i] ),
                                self.data[n]
                        )

                        + # E(a(eta_i))
                        self.exp_family.expectation_of_log_partition(
                                self.tau[i],
                                self.tau[0]
                        )

                        + # Sum of E(log(1-V_j))
                        sum(
                                [
                                        digamma( self.gamma[j,1] )
                                        - digamma( self.gamma[j,0] + self.gamma[j,1] )
                                        for j
                                        in xrange( i - 1 )
                                ]
                        )
                )
        self._normalise_phis()

    def _normalise_phis():
        #normalise phi
        for n in xrange( self.N ):
            self.phi[n,:] /= sum( self.phi[n,:] )
