#
# Copyright John Reid 2008, 2009, 2010
#


"""
Math functions for HDPM code.
"""

import numpy, logging
from scipy.special import gamma, gammaln, digamma, polygamma, betaln



def accumulate_vector(x):
    """
    @return: The accumulation of x, i.e. [x0, x0+x1, x0+x1+x2, ... , x.sum()]
    """
    result = numpy.empty_like(x)
    result[0]=x[0]
    for i in xrange(1, len(x)):
        result[i] = result[i-1]+x[i]
    return result



def array_replace(a, loc, fill):
    """
    Replace the values in a at the locations, loc, with the value, fill.
    
    For example:
    In [32]: a=numpy.arange(10)+1
    
    In [33]: array_replace(a, a%3==1, 0)
    Out[33]: array([0, 2, 3, 0, 5, 6, 0, 8, 9, 0])
    
    or for 0-d arrays:
    In [15]: b=numpy.array(3)
    
    In [16]: array_replace(b, b==3, 2)
    Out[16]: array([2])
    
    In [17]: b
    Out[17]: array(2)
    """
    a_ = numpy.atleast_1d(a)
    a_[numpy.atleast_1d(loc)] = fill
    return a_


def discrete_KL(p, q):
    "The Kullback-Leibler between 2 discrete distributions."
    l = numpy.log(p) - numpy.log(q)
    array_replace(l, p == 0., 0.)
    return (p * l).sum()


def _permutation_by_sort(a):
    argsort = a.argsort()
    N = len(a)
    result = numpy.empty_like(argsort)
    for i, a in enumerate(argsort):
        result[a] = N - i - 1
    return result

def _is_permutation(p):
    "Asserts that p is a permutation"
    return (p < len(p)).all() and (p >= 0).all()

def _permute(a, p, axis=0):
    "Permute one of a's axes according to the permutation, p. Returns an array, a is unchanged."
    assert len(p) == a.shape[axis]
    assert _is_permutation(p)
    result = numpy.empty_like(a)
    if 0 == axis:
        a_view = a
        result_view = result
    else:
        a_view = a.swapaxes(0, axis)
        result_view = result.swapaxes(0, axis)
    for i, j in enumerate(p):
        result_view[j] = a_view[i]
    return result

def _greater_than(a):
    "Return an array like the argument a, where each entry is the sum of the later entries in the original array"
    result = numpy.empty_like(a)
    result[-1] = 0.
    for k in xrange(len(a)-1):
        result[-2-k] = result[-1-k] + a[-1-k]
    return result

def _gamma_KL(a_bar, b_bar, a, b):
    "@return: KL(Gamma(a_bar,b_bar)||Gamma(a,b))"
    return (
      a_bar * numpy.log(b_bar)
      - a * numpy.log(b)
      - gammaln(a_bar)
      + gammaln(a)
      + (a_bar - a) * (digamma(a_bar) - numpy.log(b_bar))
      - a_bar * (1. - b / b_bar)
    )

def _dirichlet_KL(a_bar, a):
    "@return: KL(Dirichlet(a_bar)||Dirichlet(a))"
    a_bar_0 = a_bar.sum()
    a_0 = a.sum()
    return (
      gammaln(a_bar_0)
      - gammaln(a_0)
      - (
        gammaln(a_bar)
        - gammaln(a)
        - (a_bar - a) * (digamma(a_bar) - digamma(a_bar_0))
      ).sum()
    )

def _beta_KL(a_bar, b_bar, a, b):
    "@return: KL(Beta(a_bar, b_bar)||Beta(a, b))"
    return (
        betaln(a, b)
        - betaln(a_bar, b_bar)
        - (a - a_bar) * digamma(a_bar)
        - (b - b_bar) * digamma(b_bar)
        + (a - a_bar + b - b_bar) * digamma(a_bar + b_bar)
    )
    
def _safe_x_log_x(x):
    "Returns x log(x) with zeros when x==0."
    result = x * numpy.log(x)
    array_replace(result, x==0., 0.)
    return result

def _approx_2nd_order(f_E_plus_n, f2_E_plus_n, P_plus_n, V_plus_n):
    "Improved 2nd order approximation"
    result = numpy.asarray(P_plus_n * (f_E_plus_n + .5 * V_plus_n * f2_E_plus_n))
    array_replace(result, P_plus_n == 0., 0.)
    return result

def _approx_f_n(f, f2, P_plus_n, E_plus_n, V_plus_n):
    return _approx_2nd_order(f(E_plus_n), f2(E_plus_n), P_plus_n, V_plus_n)





class GammaDist(object):
    "p(x) proportional to x**(a-1) * exp(-bx)"

    def __init__(self, a, b):
        if a <= 0. or b <= 0.:
            raise ValueError('GammaDist: a or b not positive')

        self.a = a
        "Parameter a"

        self.b = b
        "Parameter b (a rate parameter)"

        self.E = a/b
        "Expectation"

        self.G = numpy.exp(digamma(a))/b
        "Geometric expectation"

        self.H = a - numpy.log(b) + gammaln(a) + (1.-a)*digamma(a)
        "Entropy"

    def plot(self, num_points=50, xmax=None, *plot_args, **plot_kwds):
        if xmax is None:
            xmax = 2*self.E
        x = numpy.linspace(0., xmax, num_points)
        from pylab import plot
        from scipy.stats import gamma
        plot(
            x,
            gamma.pdf(x, self.a, scale=1./self.b), 
            *plot_args, 
            **plot_kwds)

    def params(self):
        return self.a, self.b

    def __str__(self):
        return "GammaDist(%s, %s)" % self.params()

    def sample(self):
        return numpy.random.gamma(self.a, 1./self.b)

    def KL(self, other):
        assert isinstance(other, self.__class__)
        return _gamma_KL(self.a, self.b, other.a, other.b)




class BetaDist(object):
    "p(x) proportional to x**(a-1) * (1-x)**(b-1)"

    def __init__(self, a, b):
        self.a = a
        "Parameter a"

        self.b = b
        "Parameter a"

        self.E = a / (a+b)
        "Expectation"

        self.G = numpy.exp(digamma(a)-digamma(a+b))
        "Geometric expectation"
        assert numpy.isfinite(self.G).all()

        self.H = betaln(a, b)-(a-1.)*digamma(a)-(b-1.)*digamma(b)+(a+b-2)*digamma(a+b)
        "Entropy"

    def params(self):
        return self.a, self.b
    
    def sample(self):
        return numpy.random.beta(self.a, self.b)

    def KL(self, other):
        assert isinstance(other, self.__class__)
        return _beta_KL(self.a, self.b, other.a, other.b)



class DirichletDist(object):
    "p(x) proportional to prod xi**(ai-1)"

    def __init__(self, a):
        self.a = a
        "Parameter a"

        self.a_sum = a.sum()
        "Strength parameter"

        self.E = a/self.a_sum
        "Expectation"

        self.G = numpy.exp(digamma(a) - digamma(self.a_sum))
        "Geometric expectation"

        self.H = gammaln(a).sum() - gammaln(self.a_sum) + ((a-1.) * self.G).sum()
        "Entropy"

    def params(self):
        return self.a

    def sample(self):
        return numpy.random.dirichlet(self.a)

    def KL(self, other):
        assert isinstance(other, self.__class__)
        return _dirichlet_KL(self.a, other.a)



def calculate_corpus_wide_word_distribution(documents, W):
    """
    @return: The corpus wide word distribution and the number of words in the documents as a tuple.
    """
    dist = numpy.zeros((W,))
    for d in documents:
        for w in d:
            dist[w] += 1
    N = dist.sum()
    dist /= N
    return dist, N


def calculate_P_plus(E, V, Z):
    P_plus = 1. - numpy.exp(Z)
    assert (P_plus >= 0.).all()
    E_plus = E/P_plus
    V_plus = V/P_plus - numpy.exp(Z)*E_plus**2
    array_replace(E_plus, P_plus==0., 0.)
    array_replace(V_plus, P_plus==0., 0.)
    assert numpy.isfinite(P_plus).all()
    assert numpy.isfinite(E_plus).all()
    return P_plus, E_plus, V_plus


def discrete_sample(probs):
    "Sample from a discrete distribution with probabilities, probs."
    s = numpy.random.rand()
    for i, p in enumerate(probs):
        if s < p:
            return i
        s -= p
    raise RuntimeError('The probabilities sum to %f' % sum(probs))


def approximate_expected_num_tables(concentration, counts_P_plus, counts_E_plus, counts_V_plus):
    """
    Approximate the expected number of tables in a CRP.
    """
    term = concentration + counts_E_plus
    result = concentration * counts_P_plus * (
        digamma(term)
        - digamma(concentration)
        + .5 * counts_V_plus * polygamma(2, term)
    )
    # adjust for divide by zero errors when P_plus==0.
    array_replace(result, counts_P_plus == 0., 0.)
    assert numpy.isfinite(result).all()
    return result


def approximate_beta_expectation(a, counts_P_plus, counts_E_plus, counts_V_plus):
    "Approximate the expectation of the beta distributed auxiliary variable."
    term = a + counts_E_plus
    result = numpy.asarray(counts_P_plus * (digamma(a) - digamma(term) - counts_V_plus * polygamma(2, term) / 2.))
    # adjust for divide by zero errors when P_plus_n_gk==0.
    array_replace(result, counts_P_plus == 0., 0.)
    assert numpy.isfinite(result).all()
    return result


#def improved_2nd_order_approximation(f, f_double_prime, p_plus, e_plus, v_plus):
#    "@return: The improved 2nd order approximation."
#    return p_plus * (f(e_plus) + v_plus * f_double_prime(e_plus) / 2.)


