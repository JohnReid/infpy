#
# Copyright John Reid 2006
#

import scipy, math

def log_factorial( n ):
    if 0 > n: raise RuntimeError, 'n > 0: %f' % ( n )
    if 0 == n: return 0
    return math.log( n ) + log_factorial( n - 1 )

def log_n_choose( y ):
    n = sum( y )
    return (
            log_factorial( n )
            - sum( log_factorial( x ) for x in y )
    )

def log_gamma( x ):
    from scipy.special import gamma
    return math.log( gamma( x ) )

def log_uniform_Z( alpha, k ):
    return (
            k * log_gamma( alpha / k )
            - log_gamma( alpha )
    )

def log_likelihood_under_dirichlet( y ):
    k = len( y )
    if 0 == k: return 0
    n = sum( y )
    if 0 == n: return 0
    return (
            log_n_choose( y )
            - log_uniform_Z( 1.0, k )
    )

def dirichlet_rv( alpha, m ):
    from scipy.stats.distributions import gamma_gen
    gen = gamma_gen( name = 'gamma' )
    y = [ gen.rvs( m_i )[0] for m_i in m ]
    s = sum( y )
    return [ y_i / s for y_i in y ]

class MultinomialSample:
    def __init__( self, y ):
        self.y = y
        self.n = sum( y )
        self.k = len( y )
        self.log_n_choose = log_n_choose( y )
    def ll_under_multinomial( self, p ):
        return (
                self.log_n_choose
                + sum( y_i * math.log( p_i ) for p_i, y_i in zip( p, self.y ) )
        )
    def ll_under_uniform( self ):
        return self.ll_under_multinomial( [1.0 / self.k] * self.k )
    def ll_under_dirichlet( self, num_samples = 1000 ):
        result = 0
        for i in range( num_samples ):
            result += math.exp(
                    self.ll_under_multinomial(
                            dirichlet_rv( self.n, [1.0 / self.k] * self.k ) ) )
        return math.log( result / num_samples )

for y in [
        [10]*3,
        [28,1,1],
        [1,1,28],
        [16,7,7],
]:
    print y
    m = MultinomialSample( y )
    print m.ll_under_uniform()
    print m.ll_under_dirichlet(), m.ll_under_dirichlet(), m.ll_under_dirichlet()
