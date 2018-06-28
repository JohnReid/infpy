#
# Copyright John Reid 2007, 2014
#


r"""

A package that abstracts `exponential family probability distributions
<http://en.wikipedia.org/wiki/Exponential_family>`_. A probability distribution
is said to be in the exponential family if it can be represented in a specific
form:

.. math::

    p(x|\theta) = e^{\eta(\theta).T(x) - A(\theta) + h(T(x))}

Where:
    - :math:`x` is the random variable.
    - :math:`T(x)` is the random variable's sufficient statistics.
    - :math:`\theta` is the canonical parameter of the distribution,
      e.g. (mean, covariance) for a gaussian.
    - :math:`\eta(\theta)` is the natural parameter of the distribution.
    - :math:`A(\theta)` is the normalization factor (log partition function).
    - :math:`h(T(x))` is commonly 0.0 and provides a measure for :math:`x`.

The (possibly vector-valued) functions :math:`T(x)` and :math:`\eta(\theta)`
define a correspondence between the normal and canonical forms of the
distribution.

For example, if you wanted to work with a univariate Gaussian in natural form,
you could use the :class:`.GaussianExpFamily` class. Here we calculate the
log probability of drawing 0, .5 and 1 from a Gaussian distribution with
mean 0 and precision (inverse variance) 1.

.. doctest::

    >>> import infpy.exp as exp
    >>> import numpy as npy
    >>> npy.set_printoptions(precision=2)
    >>> family = exp.GaussianExpFamily()
    >>> mu = 0     # Mean
    >>> gamma = 1  # Precision (inverse variance)
    >>> theta = (mu, gamma)          # Natural parameters
    >>> eta = family.eta(theta)      # Canonical parameters
    >>> x = npy.array([0., .5, 1.])  # 3 data in natural form
    >>> T = family.T(x)              # Data in canonical form
    >>> T
    array([[ 0.  ,  0.  ],
           [ 0.5 ,  0.25],
           [ 1.  ,  1.  ]])
    >>> family.log_p_T(T, eta)       # Log probability using canonical form
    array([-0.92, -1.04, -1.42])
    >>> family.log_p_x(x, theta)     # Log probability using natural form
    array([-0.92, -1.04, -1.42])

We can also sample from exponential family distributions.

.. doctest::

    >>> npy.random.seed(1)
    >>> Tsampled = family.sample(eta, 4)  # Sample 4 sufficient statistics
    >>> Tsampled
    array([[ 1.62,  2.64],
           [-0.61,  0.37],
           [-0.53,  0.28],
           [-1.07,  1.15]])
    >>> npy.array(map(family.x, Tsampled))           # Convert to natural form
    array([ 1.62, -0.61, -0.53, -1.07])


"""


import numpy as npy
from numpy import array, asarray, dot, zeros, empty, diag
from numpy import log, exp, outer, float64, trace, sqrt, sum, linspace
from numpy.dual import solve
from numpy.linalg import det, inv, cholesky
from math import pi
from scipy.stats import norm
from scipy.special import gammaln, digamma, polygamma

_log_2_pi = log(2 * pi)


def log_multivariate_gamma(p, a):
    """ The logarithm of the `multivariate gamma
    function<http://en.wikipedia.org/wiki/Multivariate_gamma_function>`_ """

    return (p * (p - 1.) / 4.) * log(pi) + sum(gammaln((2. * a - j) / 2.)
                                               for j in xrange(p))


def log_factorial(n):
    """
    @arg n: An integer >= 0
    log n!
    """
    assert n >= 0, 'n must be >= 0: n=%d' % n
    return sum(log(i) for i in xrange(1, n + 1))


def factorial(n):
    """
    @arg n: An integer >= 0
    n!
    """
    assert n >= 0, 'n must be >= 0: n=%d' % n
    return reduce(long.__mul__, xrange(1, n + 1), 1L)


class Variable(object):

    """ An exponential family variable has a family and natural parameters for
    that family.  """

    def __init__(self, family, eta):
        "Initialise the variable."

        self.family = family
        "The exponential family that this variable belongs to."

        self.eta = eta
        "The natural parameters of this variable's distribution."


class ExponentialFamily(object):

    r"""

    An `exponential family<http://en.wikipedia.org/wiki/Exponential_family>`_
    of distributions over x.

    :math:`\log p(x|\theta) = \eta(\theta).T(x) - A(\theta) + h(T(x))`

    Where:
        - :math:`x` : the random variable
        - :math:`T(x)` : the random variable's sufficient statistics
        - :math:`\theta` : the canonical parameter of the distribution,
          e.g. (mean, covariance) for a gaussian
        - :math:`\eta(\theta)` : the natural parameter of the distribution
        - :math:`A(\theta)` : the normalization factor (log partition function)
          (can be a vector)
        - :math:`h(T)` : commonly 0.0 and provides a measure for :math:`x`

    T and eta should be numpy arrays of shape (self.dimension,).  Base classes
    that implement a specific exponential family must define T, eta, A and h
    such that the above equation holds.  They should also define some of the
    methods/attributes in the following sections.

    *Conversion* methods include:

    - theta(eta) : converts from natural parameters to canonical parameters
    - x(T) : converts from canonical parameters to natural parameters if it
      differs from 0.0
    - dimension : returns the length of eta and T
    - normalisation_dimension : returns the length of A

    The following *sampling* methods are optional:

    - sample_theta(theta, size=1) : returns a sample from the distribution
      parameterised by theta
    - sample_eta(eta, size=1) : returns a sample from the distribution
      parameterised by eta

    The following *special* methods are optional:

    - exp_T(eta) : returns :math:`d(A)/d(eta)`, the derivative of the
      normalization factor with respect to the natural parameter, evaluated at
      eta. This is equivalent to the expectation of T.

    *Testing* methods include:

    - _p_truth(x, theta) : returns :math:`p(x|\theta)` calculated by a
      different method (e.g. scipy.stats) for testing
    - _entropy_truth(theta) I{optional} : returns entropy of
      :math:`p(x|\theta)` calculated by a different method
      (e.g. scipy.stats) for testing
    - _typical_xs : a sequence of typical x values for the family
    - _typical_thetas : a sequence of typical theta values for the family
    """

    def __init__(self, dimension, normalisation_dimension=1,
                 vectorisable=False):
        """
        - dimension: The dimension of the sufficient statistics, T, and
          the natural parameter, eta
        - normalisation_dimension: The length of A(theta). Normally is one
          when a scalar is returned, can be returned as a vector of the given
          dimension. This can be useful in conjugate analysis.
        - vectorisable: Can we pass more than one T or eta to the methods of
          this family?
        """

        self.dimension = dimension
        """The dimension of the sufficient statistics, T, and the natural
        parameter, eta"""

        self.normalisation_dimension = normalisation_dimension
        """
        The length of A(theta). Normally one when a scalar is returned, can
        be returned as a vector of the given dimension. This can be useful
        in conjugate analysis.
        """

        self.vectorisable = vectorisable
        """
        Can we pass more than one T or eta to the methods of this family?
        """

    def h(self, T):
        """
        The default measure for x.

        @param T: x's sufficient statistic
        0.0
        """
        return 0.0

    def exp_h(self, eta):
        """
        The expectation of h(x) given the natural parameters, eta.
        @param eta: the natural parameters
        """
        return 0.0

    def log_p_x(self, x, theta):
        """
        @param x: the random variable in canonical form
        @param theta: the parameter in canonical form
        :math:`log p(x|\theta)`
        """
        return self.log_p_T(self.T(x), self.eta(theta))

    def log_p_T(self, T, eta):
        """
        @param T: the random variable's sufficient statistics
        @param eta: the family's parameter in natural form
        :math:`log p(T|eta)`
        """
        return dot(T, eta) - self.A(eta) + self.h(T)

    def A(self, eta):
        """
        The normalisation factor (log partition) for the exponential
        family (as a scalar).

        See A_vec() for the vectorised version.
        """
        return self.A_vec(eta).sum(axis=-1)

    def p_x(self, x, theta):
        """
        :math:`p(x|\theta)`.
        """
        return exp(self.log_p_x(x, theta))

    def p_T(self, T, eta):
        """
        :math:`p(T|eta)`.
        """
        return exp(self.log_p_T(T, eta))

    def entropy(self, eta):
        """
        The entropy of the distribution parameterised by eta. The entropy is
        calculated as the :math:`<-log p(T>|eta)>`, i.e. the expectation of
        the negative log probability.  This relies on the family defining the
        exp_T(eta) and exp_h(eta) methods.

        @arg eta: the exponential family parameters in natural form
        The entropy of the family parameterised by eta
        """
        assert hasattr(self, 'exp_T'), 'Subclass does not define exp_T'
        assert hasattr(self, 'exp_h'), 'Subclass does not define exp_h'
        return -dot(self.exp_T(eta), eta) + self.A(eta) - self.exp_h(eta)

    def KL(self, eta_1, eta_2):
        """
        Returns the Kullback-Leibler divergence between the the distributions
        in this family parameterised by eta_1 and eta_2.
        @arg eta_1: Natural parameters.
        @arg eta_2: Natural parameters.
        KL(eta_1||eta_2) = E[log p(x|eta_1)] - E[log p(x|eta_2)]
        where expectations are w.r.t. p(x|eta_1)
        """
        assert hasattr(self, 'exp_T'), 'Subclass does not define exp_T'
        return (dot(self.exp_T(eta_1), eta_1 - eta_2)
                - self.A(eta_1) + self.A(eta_2))

    def _check_shape(self, arg):
        """
        True if the argument, for example T or eta, is the correct
        shape.
        """
        if self.vectorisable:
            return arg.shape[-1] == self.dimension
        else:
            return arg.shape == (self.dimension,)

    def _empty(self):
        """
        An empty array of the correct size for T or eta.
        """
        return empty((self.dimension,), dtype=float64)

    def LL_fns(self, tau, nu):
        """
        ll, ll_prime, ll_hessian : The log likelihood function and its
        derivative and hessian for the given tau and nu prior.
        """
        def ll(eta):
            return dot(eta, tau) - nu * self.A(eta)

        def ll_prime(eta):
            return tau - nu * self.exp_T(eta)

        def ll_hess(eta):
            return - nu * self.cov_T(eta)

        return ll, ll_prime, ll_hess


class GaussianExpFamily(ExponentialFamily):

    r"""
    The univariate gaussian distribution in exponential family form.

    - :math:`\theta = (\mu, \gamma)` where :math:`\mu` is the mean and
      :math:`\gamma` is the precision
    - :math:`T = (x, x^2)`
    - :math:`\eta = (\mu\gamma, -\frac{\gamma}{2})`
    - :math:`A = -\frac{1}{2}(\log\frac{\gamma}{2\pi} - \gamma \mu^2)`

    """

    def __init__(self):
        """
        Initialise this exponential family.
        """
        ExponentialFamily.__init__(self, 2, normalisation_dimension=2)

    _typical_xs = [
        0,
        -10.0,
        5.0,
    ]
    "A sequence of typical x values for this family."

    _typical_thetas = [
        (0.0, 1.0),
        (0.0, .5),
        (10.0, .1),
        (0.0, 10.0),
        (-10.0, .001),
        (100.0, 4.),
    ]
    "A sequence of typical theta values for this family."

    def h(self, T):
        """
        The default measure for x.

        @param T: x's sufficient statistic
        0.0
        """
        return - .5 * _log_2_pi

    def exp_h(self, eta):
        """
        The expectation of h(x) given the natural parameters, eta.
        @param eta: the natural parameters
        """
        return - .5 * _log_2_pi

    def T(self, x):
        "T(x), the sufficient statistics of x"
        return array([x, x * x]).T

    def exp_T(self, eta):
        """
        @arg eta: The natural parameters.
        The expectation of T, the sufficient statistics, given eta.
        """
        (mu, gamma) = self.theta(eta)
        return array([mu, mu ** 2 + 1. / gamma])

    def x(self, T):
        "x(T), the x that has the sufficient statistics, T"
        # assert self._check_shape(
        #     T), 'T does not have the correct shape: %s' % str(T.shape)
        return npy.take(T, 0, -1)

    def eta(self, theta):
        r""":math:`\eta(\theta)`, the natural parameter :math:`\eta`, that
        corresponds to the canonical parameter, :math:`\theta`.
        """
        theta = asarray(theta, dtype=float)
        return array([theta[0] * theta[1], -theta[1] / 2])

    def theta(self, eta):
        """The canonical parameter, theta, that
        corresponds to the natural parameter, eta"""
        assert self._check_shape(eta)
        gamma = -2 * eta[1]
        return (eta[0] / gamma, gamma)

    def A_vec(self, eta):
        "The normalization factor (log partition)"
        (mu, gamma) = self.theta(eta)
        return .5 * array([-log(gamma), gamma * mu * mu])

    def sample(self, eta, size=1):
        """
        @param eta: the natural parameters
        @param size: the size of the sample
        A sample of sufficient statistics
        """
        (mu, gamma) = self.theta(eta)
        return self.T(norm.rvs(size=size, loc=mu, scale=1 / sqrt(gamma)))

    def _p_truth(self, x, theta):
        """
        :math:`log p(x|\theta)` calculated by an independent method.
        (Primarily for testing purposes)
        """
        (mu, gamma) = theta
        return norm.pdf(x, loc=mu, scale=1 / sqrt(gamma))

    def _entropy_truth(self, theta):
        """
        entropy of :math:`p(x|\theta)` calculated by an independent
        method. (Primarily for testing purposes)
        """
        (_mu, gamma) = theta
        return .5 * (1. + _log_2_pi - log(gamma))


class DirichletExpFamily(ExponentialFamily):

    """
    The Dirichlet distribution in exponential family form.

        - k: the number of outcomes
        - :math:`x = p_1, \dots, p_k`
        - :math:`T(x) = \log x`
        - :math:`\eta(\theta) = \theta-1`

    Does not handle the case where x = 0.
    """

    def __init__(self, k=2):
        """
        Initialise the Dirichlet exponential family.
        """
        super(type(self), self).__init__(k, vectorisable=True)

    _typical_xs = array([
        [.5, .5],
        [1e-3, 1. - 1e-3],
    ])
    "A sequence of typical x values for this family."

    _typical_thetas = array([
        (.5, .5),
        (.05, .95),
        # (.0, 1.), # Cannot get this to work - trouble with log(0)'s
        (50.0, 50.0),
        (99.0, 1.0),
    ])
    "A sequence of typical theta values for this family."

    def T(self, x):
        "T(x), the sufficient statistics of x"
        return log(x)

    def exp_T(self, eta):
        """
        @arg eta: The natural parameters.
        The expectation of T, the sufficient statistics, given eta.
        """
        theta = self.theta(eta)
        return (digamma(theta)
                - digamma(theta.sum(axis=-1)).reshape(theta.shape[:-1] + (1,)))

    def cov_T(self, eta):
        """
        @arg eta: The natural parameters.
        The covariance of T_i, T_j, the sufficient statistics, given
        eta.
        """
        theta = self.theta(eta)
        assert (self.dimension,) == theta.shape
        return diag(polygamma(1, theta)) - polygamma(1, theta.sum())

    def x(self, T):
        "x(T), the x that has the sufficient statistics, T"
        assert self._check_shape(
            T), 'T does not have the correct shape: %s' % str(T.shape)
        return exp(T)

    def eta(self, theta):
        """The natural parameter, eta, that corresponds to
        the canonical parameter, theta"""
        return theta - 1.

    def theta(self, eta):
        """The canonical parameter, theta, that corresponds
        to the natural parameter, eta"""
        assert self._check_shape(eta)
        return eta + 1.

    def A_vec(self, eta):
        "The normalization factor (log partition)"
        return (gammaln(eta + 1).sum(axis=-1)
                - gammaln(eta.sum(axis=-1)
                          + self.dimension)).reshape(eta.shape[:-1] + (1,))

    def sample(self, eta, size=1):
        """
        @param eta: the natural parameters
        @param size: the size of the sample
        A sample of sufficient statistics
        """
        from numpy.random import dirichlet
        return self.T(dirichlet(self.theta(eta), size=size))

    def _p_truth(self, x, theta):
        """
        :math:`log p(x|\theta)` calculated by an independent method.
        (Primarily for testing purposes)
        """
        assert len(x) == len(theta)
        return exp(
            ((theta - 1.) * log(x) - gammaln(theta)).sum()
            + gammaln(sum(theta))
        )

    def plot(self, eta, scale=1., *plot_args, **plot_kwds):
        import pylab as pl
        if 2 == self.dimension:
            num_steps = 1001
            r = empty((num_steps, 2.))
            r[:, 0] = linspace(0., 1., num_steps)
            r[:, 1] = 1. - r[:, 0]
            theta = self.theta(eta)
            p = scale * self.p_x(r, theta)
            if 'label' not in plot_kwds:
                plot_kwds['label'] = 'Theta = %s' % str(theta)
            pl.plot(r[:, 0], p, *plot_args, **plot_kwds)
            return r[:, 0], p


class GammaExpFamily(ExponentialFamily):

    """ The `gamma
    distribution<http://en.wikipedia.org/wiki/Gamma_distribution>`_ in
    exponential family form.

            - theta = (a,b) where a is the shape and b is the
              rate (inverse scale)
            - eta = (-b, a-1)
            - T(x) = (x, log(x))
            - A(theta) = log Gamma(a) - a log(b)
    """

    _typical_xs = [
        .1,
        5.,
    ]
    "A sequence of typical x values for this family."

    _typical_thetas = [
        (.1, 1.0),
        (10.0, .1),
        (0.1, 10.0),
        (100.0, 4.),
    ]
    "A sequence of typical theta values for this family."

    def __init__(self):
        """
        Initialise
        """
        ExponentialFamily.__init__(self, 2)

    def T(self, x):
        "T(x), the sufficient statistics of x"
        return array((x, log(x))).T

    def exp_T(self, eta):
        """
        @arg eta: The natural parameters.
        The expectation of T, the sufficient statistics, given eta.
        """
        (a, b) = self.theta(eta)
        return array((a / b, digamma(a) - log(b)))

    def x(self, T):
        "x(T), the x that has the sufficient statistics, T"
        assert self._check_shape(
            T), 'T does not have the correct shape: %s' % str(T.shape)
        return T[0]

    def eta(self, theta):
        """The natural parameter, eta, that corresponds to
        the canonical parameter, theta"""
        (a, b) = theta
        return array((-b, a - 1.))

    def theta(self, eta):
        """The canonical parameter, theta, that corresponds
        to the natural parameter, eta"""
        assert self._check_shape(eta)
        return array((eta[1] + 1, -eta[0]))

    def A_vec(self, eta):
        "The normalization factor (log partition)"
        (a, b) = self.theta(eta)
        return array((gammaln(a) - a * log(b),))

    def sample(self, eta, size=1):
        """
        @param eta: the natural parameters
        @param size: the size of the sample
        A sample of sufficient statistics
        """
        from numpy.random import gamma
        (a, b) = self.theta(eta)
        return self.T(gamma(a, scale=1. / b, size=size))

    def _p_truth(self, x, theta):
        """
        :math:`log p(x|\theta)` calculated by an independent method.
        (Primarily for testing purposes)
        """
        from scipy.stats import gamma
        (a, b) = theta
        return gamma.pdf(
            x,
            a,
            scale=1. / b
        )

    def _entropy_truth(self, theta):
        """
        entropy of :math:`p(x|\theta)` calculated by an independent
        method. (Primarily for testing purposes)
        """
        (a, b) = theta
        return a - log(b) + gammaln(a) + (1. - a) * digamma(a)


class PoissonExpFamily(ExponentialFamily):

    """ The `Poisson
    distribution<http://en.wikipedia.org/wiki/Poisson_distribution>`_ in
    exponential family form.

            - theta = lambda where lambda is the rate parameter
            - eta = log lambda
            - T(x) = x
            - A(theta) = lambda
            - h(x) = -log x!
    """

    _typical_xs = [
        1,
        3,
        10
    ]
    "A sequence of typical x values for this family."

    _typical_thetas = [
        1,
        4,
        10,
    ]
    "A sequence of typical theta values for this family."

    def __init__(self):
        """
        Initialise.
        """
        ExponentialFamily.__init__(self, 1)

    def T(self, x):
        "T(x), the sufficient statistics of x"
        return asarray([x])

    def exp_T(self, eta):
        """
        @arg eta: The natural parameters.
        The expectation of T, the sufficient statistics, given eta.
        """
        _lambda = self.theta(eta)
        return self.T(_lambda)

    def x(self, T):
        "x(T), the x that has the sufficient statistics, T"
        assert self._check_shape(
            T), 'T does not have the correct shape: %s' % str(T.shape)
        return T[0]

    def eta(self, theta):
        """The natural parameter, eta, that corresponds
        to the canonical parameter, theta"""
        _lambda = theta
        return log(asarray([_lambda]))

    def theta(self, eta):
        """The canonical parameter, theta, that
        corresponds to the natural parameter, eta"""
        assert self._check_shape(eta)
        return exp(eta[0])

    def A_vec(self, eta):
        "The normalization factor (log partition)"
        _lambda = self.theta(eta)
        return array((_lambda,))

    def h(self, T):
        """
        The measure for x in this exponential family.

        @param T: x's sufficient statistic
        """
        return -log_factorial(int(self.x(T)))

    def exp_h(self, eta):
        """
        The expectation of h(x) given the natural parameters, eta.
        @param eta: the natural parameters
        """
        _lambda = self.theta(eta)
        result = 0.0
        eps = 1.e-18
        k = 1
        log_term = 0.
        factorial_term = 1.
        while True:
            log_term += log(k)
            factorial_term *= k
            term = _lambda ** k * log_term / factorial_term
            if abs(term) < abs(result * eps):
                break
            result += term
            k += 1
        return -result * exp(-_lambda)

    def sample(self, eta, size=1):
        """
        @param eta: the natural parameters
        @param size: the size of the sample
        A sample of sufficient statistics
        """
        from numpy.random import poisson
        _lambda = self.theta(eta)
        return poisson(lam=_lambda, size=size).reshape((size, 1))

    def _p_truth(self, x, theta):
        """
        :math:`log p(x|\theta)` calculated by an independent method.
        (Primarily for testing purposes)
        """
        _lambda = theta
        return _lambda ** x * exp(-_lambda) / factorial(x)


class DiscreteExpFamily(ExponentialFamily):

    """
    The discrete distribution in exponential family form. Also known as
    the multinomial distribution.

            - T(x) = delta(x=i)
            - theta = (p1, ..., pk)
            - eta = log(theta)
            - A(theta) = 0
    """

    def __init__(self, k=2):
        """
        Initialise the discrete distribution.
        @arg k: number of possible outcomes
        """
        ExponentialFamily.__init__(self, dimension=k,
                                   normalisation_dimension=0)

    _typical_xs = [
        0,
        1,
    ]
    "A sequence of typical x values for this family."

    _typical_thetas = [
        (.1, .9),
        (.5, .5),
    ]
    "A sequence of typical theta values for this family."

    def T(self, x):
        "T(x), the sufficient statistics of x"
        T = zeros(self.dimension)
        T[x] = 1.0
        return T

    def exp_T(self, eta):
        """
        @arg eta: The natural parameters.
        The expectation of T, the sufficient statistics, given eta.
        """
        return self.theta(eta)

    def x(self, T):
        "x(T), the x that has the sufficient statistics, T"
        assert self._check_shape(
            T), 'T does not have the correct shape: %s' % str(T.shape)
        for x in xrange(self.dimension):
            if T[x]:
                return x
        else:
            raise RuntimeError("Could not find positive value in u")

    def eta(self, theta):
        """The natural parameter, eta, that corresponds
        to the canonical parameter, theta"""
        return log(theta)

    def theta(self, eta):
        """The canonical parameter, theta, that
        theta(eta), t
        corresponds to the natural parameter, eta"""
        assert self._check_shape(eta)
        return exp(eta)

    def A_vec(self, eta):
        "The normalization factor (log partition)"
        return zeros((0,))

    def sample(self, eta, size=1):
        """
        @param eta: the natural parameters
        @param size: the size of the sample
        A sample of sufficient statistics
        """
        from numpy.random import multinomial
        from itertools import chain, repeat
        multi_sample = multinomial(n=size, pvals=self.theta(eta))
        return array([self.T(x)
                      for x
                      in chain(*[repeat(i, count)
                                 for i, count
                                 in enumerate(multi_sample)])])

    def _p_truth(self, x, theta):
        """
        :math:`log p(x|\theta)` calculated by an independent method.
        (Primarily for testing purposes)
        """
        return theta[x]

    def _entropy_truth(self, theta):
        """
        entropy of :math:`p(x|\theta)` calculated by an independent
        method. (Primarily for testing purposes)
        """
        return - dot(theta, log(theta))


class MvnExpFamily(ExponentialFamily):

    r""" The `multi-variate normal
    distribution<http://en.wikipedia.org/wiki/Multivariate_normal_distribution>`_
    in k dimensions in exponential family form.

    - k: the number of dimensions

    Exponential family parameterisation:

    - :math:`T(x) = (x, x x^t)`
    - :math:`\theta = (mu,W)` where :math:`\mu` is the mean and :math:`W`
      is the precision
    - :math:`\eta(\theta) = (W \mu, \frac{W}{2})`
    - :math:`A(\theta) = \frac{1}{2} [\mu W \mu - \log|W| + k \log(2\pi)]`
    """

    def __init__(self, k=2):
        """Initialise this exponential family."""
        ExponentialFamily.__init__(self, k * (k + 1), 2)

        self.k = k
        "The number of dimensions of x."

    _typical_xs = array((
        (0., 0.),
        (-1., 1.),
        (2., 2.),
    ))
    "A sequence of typical x values for this family."

    _typical_thetas = [
        (
            array([0.0, 0.0]),
            array([[1., 0.], [0., 1.]])
        ),
        (
            array([1.0, -2.0]),
            array([[.1, 0.], [0., 2.]])
        ),
        (
            array([1.0, -2.0]),
            array([[.6, .3], [.3, .6]])
        ),
    ]
    "A sequence of typical theta values for this family."

    def T(self, x):
        "The sufficient statistics of x."
        x = asarray(x)
        T = self._empty()
        T[:self.k] = x
        T[self.k:] = -outer(x, x).reshape((self.k ** 2,))
        return T

    def exp_T(self, eta):
        """The expectation of T, the sufficient statistics, given eta."""
        (mu, W) = self._mu_W_from_theta(self.theta(eta))
        exp_T = self._empty()
        exp_T[:self.k] = mu
        exp_T[self.k:] = -(inv(W) + outer(mu, mu)).reshape((self.k ** 2,))
        return exp_T

    def x(self, T):
        "x(T), the x that has the sufficient statistics, T."
        assert self._check_shape(
            T), 'T does not have the correct shape: %s' % str(T.shape)
        return T[:self.k]

    def eta(self, theta):
        """The natural parameter, eta, that corresponds to
        the canonical parameter, theta."""
        (mu, W) = self._mu_W_from_theta(theta)
        eta = self._empty()
        eta[:self.k] = dot(W, mu)
        eta[self.k:] = .5 * W.reshape((self.k ** 2,))
        return eta

    def theta(self, eta):
        """The canonical parameter, theta, that
        corresponds to the natural parameter, eta."""
        assert self._check_shape(eta)
        W = 2. * eta[self.k:].reshape((self.k, self.k))
        mu = solve(W, eta[:self.k])
        return (mu, W)

    def A_vec(self, eta):
        "The normalization factor (log partition)."
        (mu, W) = self.theta(eta)
        return array([
            .5 * (self.k * _log_2_pi - log(det(W))),
            .5 * dot(dot(mu.T, W), mu)
        ])

    def sample(self, eta, size=1):
        """
        A sample of sufficient statistics of the given size given
        the natural parameters, eta.
        """
        from numpy.random import multivariate_normal
        mu, W = self._mu_W_from_theta(self.theta(eta))
        return array([self.T(x)
                      for x
                      in multivariate_normal(mu, inv(W), [size])])

    def _mu_W_from_theta(self, theta):
        """Helper method to extract mu and W from theta and ensure are
        correct type and shape."""
        (mu, W) = theta
        mu = asarray(mu)
        W = asarray(W)
        assert mu.shape == (self.k,)
        assert W.shape == (self.k, self.k)
        return (mu, W)

    def _p_truth(self, x, theta):
        """
        :math:`log p(x|\theta)` calculated by an independent method.
        (Primarily for testing purposes)
        """
        (mu, W) = self._mu_W_from_theta(theta)
        x_ = asarray(x) - mu
        return exp(.5 * (log(det(W))
                         - self.k * _log_2_pi
                         - dot(dot(x_.T, W), x_)))


class WishartExpFamily(ExponentialFamily):

    r"""
    The `Wishart distribution
    <http://en.wikipedia.org/wiki/Wishart_distribution>`_ in p dimensions
    in exponential family form.

    .. math::

        X \sim W_p(V, n)

    where:

        - :math:`n > p - 1` : degrees of freedom
        - :math:`V` : scale matrix (positive definite)

    Exponential family parameterisation:

    - :math:`T(W) = [\log |W|, W]` (flattened)
    - :math:`\theta = (n, V)`
    - :math:`\eta(\theta) = \frac{1}{2}[n-p-1, -V^{-1}]` (flattened)
    - :math:`A(\theta) = \frac{1}{2}(n p \log 2 + n \log |V|)
      + \log \Gamma_p (\frac{n}{2})`

    where :math:`\Gamma_p` is the `multivariate gamma distribution
    <http://en.wikipedia.org/wiki/Multivariate_gamma_function>`_.

    """

    _typical_xs = [
        array([[1., 0.], [0., 1.]]),
        array([[2, -.3], [-.3, 4.]]),
    ]
    "A sequence of typical x values for this family."

    _typical_thetas = [
        (
            3.,
            array([[1., 0.], [0., 1.]])
        ),
        (
            3.,
            array([[1., .3], [.3, 1.]])
        ),
    ]
    "A sequence of typical theta values for this family."

    def __init__(self, p=2):
        """Initialise this exponential family."""
        ExponentialFamily.__init__(self, p ** 2 + 1)

        self.p = p
        "The number of dimensions."

    def T(self, x):
        "T(x), the sufficient statistics of x."
        x = asarray(x)
        T = self._empty()
        T[0] = log(det(x))
        T[1:] = x.reshape(self.p ** 2,)
        return T

    def exp_T(self, eta):
        """
        The expectation of T, the sufficient statistics, given
        the natural parameters, eta.
        """
        assert self._check_shape(eta)
        n, V = self.theta(eta)
        T = self._empty()
        T[0] = self.p * log(2.) + log(det(V)) + \
            sum(digamma((n - i) / 2.) for i in xrange(self.p))
        T[1:] = (n * V).reshape(self.p ** 2,)
        return T

    def x(self, T):
        "x(T), the x that has the sufficient statistics, T."
        assert self._check_shape(
            T), 'T is not the correct shape: %s' % str(T.shape)
        return asarray(T[1:].reshape((self.p, self.p)))

    def eta(self, theta):
        """The natural parameter, eta, that corresponds
        to the canonical parameter, theta."""
        (n, V) = self._n_V_from_theta(theta)
        eta = self._empty()
        eta[0] = n - self.p - 1.
        eta[1:] = (-inv(V)).reshape((self.p ** 2,))
        eta /= 2.
        return eta

    def theta(self, eta):
        """The canonical parameter, theta, that
        corresponds to the natural parameter, eta."""
        assert self._check_shape(eta)
        n = 2. * eta[0] + 1. + self.p
        V = -inv(2. * eta[1:].reshape((self.p, self.p)))
        return (n, V)

    def A_vec(self, eta):
        "The normalization factor (log partition)."
        n, V = self.theta(eta)
        return array((.5 * (
            n * (self.p * log(2.) + log(det(V)))
        ) + log_multivariate_gamma(self.p, n / 2.),))

    def sample(self, eta, size=1):
        """
        A sample of sufficient statistics given the natural parameters, eta.

        This uses the method detailed by `Smith & Hocking
        <http://en.wikipedia.org/wiki/Wishart_distribution#Drawing_values_from_the_distribution>`_.
        """
        from scipy.stats import norm, chi2
        X = empty((size, self.dimension), float64)
        n, V = self.theta(eta)
        L = cholesky(V)
        std_norm = norm(0, 1)
        for sample_idx in xrange(size):
            # avoid singular matrices by resampling until the determinant is !=
            # 0.0
            while True:
                A = zeros((self.p, self.p), dtype=float64)
                for i in xrange(self.p):
                    A[i, :i] = std_norm.rvs(size=i)
                    A[i, i] = sqrt(chi2.rvs(n - i))
                if det(A) != 0.0:
                    break
            X[sample_idx] = self.T(dot(L, dot(A, dot(A.T, L.T))))
        return X

    def exp_log_det_W(self, eta):
        r"""
        Calculate :math:`\log det(W)` given the natural parameters, eta.
        """
        n, V = self.theta(eta)
        return (self.p * log(2.)
                + log(det(V))
                + sum(digamma((n - i) / 2.) for i in xrange(self.p)))

    def _n_V_from_theta(self, theta):
        """Helper method to extract n and V from theta and ensure are
        correct type and shape."""
        (n, V) = theta
        V = asarray(V)
        assert V.shape == (self.p, self.p)
        return (n, V)

    def _p_truth(self, x, theta):
        r"""
        :math:`log p(x|\theta)` calculated by an independent method.
        (Primarily for testing purposes)
        """
        n, V = self._n_V_from_theta(theta)
        x = asarray(x)
        return (
            det(x) ** ((n - self.p - 1.) / 2.)
            * exp(-trace(dot(inv(V), x)) / 2.)
            * 2. ** (-n * self.p / 2.)
            * det(V) ** (-n / 2.)
            / exp(log_multivariate_gamma(self.p, n / 2.))
        )


class NormalGammaExpFamily(ExponentialFamily):

    r"""
    A Normal-Gamma distribution in 1 dimension.
    Univariate version of Normal-Wishart multivariate.
    This is a conjugate prior for the univariate normal (Gaussian)
    distribution.

    - :math:`\gamma|\alpha, \beta \sim \Gamma(\alpha, \beta)`
    - :math:`\mu|\gamma, \mu_0, \lambda \sim
      \mathcal{N}(\mu_0, \frac{\lambda}{\gamma})`

    where the parameters are:

    - :math:`\mu_0` : mean
    - :math:`\lambda` : scaling factor for variance
    - :math:`\alpha` : shape of gamma prior on precision
    - :math:`\beta` : rate of gamma prior on precision

    The exponential family parameterisation:

    - :math:`x = (\mu, \gamma)`
    - :math:`T(x) = [\log \gamma, \mu^2 \gamma, \mu \gamma, -\frac{\gamma}{2}]`
    - :math:`\theta = (\alpha, \beta, \mu_0, \lambda)`
    - :math:`\eta(\theta) = [\alpha-\frac{1}{2}, \frac{1}{2 \lambda},
      \frac{\mu_0}{\lambda}, 2 \beta + \frac{\mu_0^2}{\lambda}]`
    - :math:`A(\theta) = \alpha \log \beta + \log \Gamma(\alpha)
      - \frac{1}{2}\log(2 \pi \lambda)` """

    _typical_xs = [
        [0., 1.],
        [0., 2.],
        [1., 1.],
        [-1., .6],
        [10., 3.],
    ]
    "A sequence of typical x values for this family."

    _typical_thetas = [
        (
            1.,
            1.,
            0.,
            1.
        ),
        (
            2.,
            3.,
            -2.,
            4.
        ),
    ]
    "A sequence of typical theta values for this family."

    def __init__(self, p=2):
        """
        Initialise this exponential family.
        @arg p: the number of dimensions
        """
        self.gamma = GammaExpFamily()
        "The Gamma distribution"

        self.gaussian = GaussianExpFamily()
        "The Gaussian distribution"

        ExponentialFamily.__init__(self, 4)

    def T(self, x):
        "T(x), the sufficient statistics of x"
        # [log gamma, mu**2 gamma, mu*gamma, -gamma/2]
        mu, gamma = x
        T = self._empty()
        T[0] = .5 * log(gamma)
        T[1] = -.5 * mu ** 2 * gamma
        T[2] = mu * gamma
        T[3] = -gamma / 2.
        return T

    def exp_T(self, eta):
        """
        @arg eta: The natural parameters.
        The expectation of T, the sufficient statistics, given eta.
        """
        alpha, beta, mu_0, _lambda = self.theta(eta)
        ab_ratio = alpha / beta
        T = self._empty()
        T[0] = .5 * (digamma(alpha) - log(beta))
        T[1] = -.5 * (_lambda + ab_ratio * mu_0 ** 2)
        T[2] = mu_0 * ab_ratio
        T[3] = - ab_ratio / 2.
        return T

    def x(self, T):
        "x(T), the x that has the sufficient statistics, T"
        assert self._check_shape(
            T), 'T is not the correct shape: %s' % str(T.shape)
        gamma = -2. * T[3]
        mu = T[2] / gamma
        return array((mu, gamma))

    def eta(self, theta):
        """The natural parameter, eta, that
        corresponds to the canonical parameter, theta"""
        alpha, beta, mu_0, _lambda = theta
        eta = self._empty()
        eta[0] = 2. * alpha - 1.
        eta[1] = 1 / _lambda
        eta[2] = mu_0 / _lambda
        eta[3] = 2. * beta + mu_0 ** 2 / _lambda
        return eta

    def theta(self, eta):
        """The canonical parameter, theta, that
        corresponds to the natural parameter, eta"""
        assert self._check_shape(eta)
        alpha = (eta[0] + 1.) / 2.
        _lambda = 1. / eta[1]
        mu_0 = eta[2] * _lambda
        beta = (eta[3] - mu_0 ** 2 / _lambda) / 2.
        return (alpha, beta, mu_0, _lambda)

    def A_vec(self, eta):
        "The normalization factor (log partition)"
        alpha, beta, _mu_0, _lambda = self.theta(eta)
        return array((
            gammaln(alpha)
            + (_log_2_pi + log(_lambda)) / 2.
            - alpha * log(beta),
        ))

    def sample(self, eta, size=1):
        """
        @param eta: the natural parameters
        @param size: the size of the sample
        A sample of sufficient statistics
        """
        # from IPython.Debugger import Pdb; Pdb().set_trace()
        samples = empty((size, self.dimension))
        alpha, beta, mu_0, _lambda = self.theta(eta)
        gammas = [self.gamma.x(gamma) for gamma in self.gamma.sample(
            self.gamma.eta((alpha, beta)), size=size)]
        assert len(gammas) == size
        for i, gamma in enumerate(gammas):
            mu = self.gaussian.x(
                self.gaussian.sample(eta=self.gaussian.eta((mu_0,
                                                            gamma / _lambda)),
                                     size=1)[0])
            samples[i] = self.T((mu, gamma))
        assert len(samples) == size
        return samples

    def _p_truth(self, x, theta):
        """
        :math:`log p(x|\theta)` calculated by an independent method.
        (Primarily for testing purposes)
        """
        alpha, beta, mu_0, _lambda = theta
        mu, gamma = x
        return (self.gamma._p_truth(gamma, (alpha, beta))
                * self.gaussian._p_truth(mu, (mu_0, gamma / _lambda)))


class NormalWishartExpFamily(ExponentialFamily):

    r"""
    A Normal-Wishart distribution in p dimensions.
    This is a conjugate prior for the multivariate normal distribution.

     - :math:`W \sim \textrm{Wishart}(\nu,S)`
     - :math:`\mu|W \sim \mathcal{N}(\mu_0, \frac{W^{-1}}{\kappa_0})`

    where the parameters are:

     - :math:`\nu` : degrees of freedom of precision
     - :math:`S` : precision
     - :math:`\mu_0` : mean
     - :math:`\kappa_0` : prior strength

    The exponential family parameterisation:

     - :math:`x = (\mu, W)`
     - :math:`T(x) = [\frac{1}{2}(\log|W|-p\log(2\pi)),
       -\frac{1}{2}\mu' W \mu, \eta_\textrm{MVN}(x)]`
     - :math:`\theta = (\nu, S, \kappa_0, \mu_0)`
     - :math:`\eta(\theta) = [\kappa_0, \nu-p, -\kappa_0.\mu_0,
       \kappa_0 \mu_0 \mu_0'-S^{-1}]`
     - :math:`A(\theta) = \frac{p}{2}[(p+1)\log 2
       - (\nu-p-1)\log(\pi) - \log \kappa_0]
       + \frac{\nu}{2} \log|S| + \log \Gamma_p(\frac{\nu}{2})`

    where :math:`\eta_\textrm{MVN}(x)` is the :math:`\eta()` function from the
    MVN exponential family and :math:`\Gamma_p` is the `multivariate gamma
    distribution <http://en.wikipedia.org/wiki/Multivariate_gamma_function>`_.

    """

    _typical_xs = [
        (array([0., 0.]), array([[1., 0.], [0., 1.]])),
        (array([-1., .3]), array([[.8, .2], [.2, .7]])),
    ]
    "A sequence of typical x values for this family."

    _typical_thetas = [
        (
            3.,
            array([[1., 0.], [0., 1.]]),
            1.,
            array([0., 0.]),
        ),
        (
            4.,
            array([[1., .3], [.3, 1.2]]),
            1.,
            array([-2., 1.]),
        ),
    ]
    "A sequence of typical theta values for this family."

    def __init__(self, p=2):
        """
        Initialise this exponential family.
        @arg p: the number of dimensions
        """
        self.wishart = WishartExpFamily(p)
        "The Wishart distribution"

        self.mvn = MvnExpFamily(p)
        "The Normal distribution"

        ExponentialFamily.__init__(self, self.mvn.dimension + 2)

        self.p = p
        "The dimension of mu"

    def T(self, x):
        "The sufficient statistics of x"
        mu, W = x
        T = self._empty()
        T[2:] = self.mvn.eta(x)
        T[0] = (log(det(W)) - self.p * _log_2_pi) / 2.
        T[1] = - dot(T[2:2 + self.p], mu) / 2.
        return T

    def exp_T(self, eta):
        """
        The expectation of T, the sufficient statistics, given the natural
        parameters, eta.
        """
        nu, S, _kappa_0, mu_0 = self.theta(eta)
        T = self._empty()
        T[0] = (log(det(S)) + sum(digamma((nu - i) / 2.)
                                  for i
                                  in xrange(self.p)) - self.p * log(pi)) / 2.
        T[2:] = self.mvn.eta((mu_0, nu * S))
        T[1] = - dot(mu_0, T[2:2 + self.p]) / 2.
        return T

    def x(self, T):
        "The x that has the sufficient statistics, T"
        assert self._check_shape(
            T), 'T is not the correct shape: %s' % str(T.shape)
        return self.mvn.theta(T[2:])

    def eta(self, theta):
        """The natural parameter, eta, that
        corresponds to the canonical parameter, theta"""
        nu, S, kappa_0, mu_0 = self._unpack_theta(theta)
        eta = self._empty()
        eta[0] = nu - self.p
        eta[1] = kappa_0
        eta[2:2 + self.p] = kappa_0 * mu_0
        eta[2 + self.p:] = - \
            (kappa_0 * outer(mu_0, mu_0) + inv(S)).reshape((self.p ** 2,))
        return eta

    def theta(self, eta):
        """The canonical parameter, theta, that
        corresponds to the natural parameter, eta"""
        assert self._check_shape(eta)
        nu = eta[0] + self.p
        kappa_0 = eta[1]
        mu_0 = eta[2:2 + self.p] / kappa_0
        S = -inv(kappa_0 * outer(mu_0, mu_0) +
                 eta[2 + self.p:].reshape((self.p, self.p)))
        return (nu, S, kappa_0, mu_0)

    def A_vec(self, eta):
        "The normalization factor (log partition)"
        nu, S, kappa_0, _mu_0 = self.theta(eta)
        return array(((
            ((self.p + 1.) * log(2.) - (nu - self.p - 1.)
             * log(pi) - log(kappa_0)) * self.p / 2.
            + log(det(S)) * nu / 2.
            + log_multivariate_gamma(self.p, nu / 2.)
        ),))

    def sample(self, eta, size=1):
        """
        A sample of sufficient statistics given the natural parameters, eta.
        """
        # from IPython.Debugger import Pdb; Pdb().set_trace()
        samples = empty((size, self.dimension))
        nu, S, kappa_0, mu_0 = self.theta(eta)
        Ws = [self.wishart.x(W) for W in self.wishart.sample(
            self.wishart.eta((nu, S)), size=size)]
        assert len(Ws) == size
        for i, W in enumerate(Ws):
            mu = self.mvn.x(
                self.mvn.sample(eta=self.mvn.eta((mu_0, W * kappa_0)),
                                size=1)[0])
            samples[i] = self.T((mu, W))
        assert len(samples) == size
        return samples

    def _unpack_theta(self, theta):
        "Extract components of theta and check/correct their shapes."
        (nu, S, kappa_0, mu_0) = theta

        S = asarray(S)
        assert S.shape == (self.p, self.p)

        mu_0 = asarray(mu_0)
        assert mu_0.shape == (self.p,)

        return (nu, S, kappa_0, mu_0)

    def _p_truth(self, x, theta):
        """
        :math:`log p(x|\theta)` calculated by an independent method.
        (Primarily for testing purposes)
        """
        nu, S, kappa_0, mu_0 = self._unpack_theta(theta)
        mu, W = x
        mu = asarray(mu)
        W = asarray(W)
        return (self.wishart._p_truth(W, (nu, S))
                * self.mvn._p_truth(mu, (mu_0, W * kappa_0)))


class ConjugatePrior(object):

    """ Represents a conjugate prior relationship between 2 exponential
    families, one for the likelihood and one for the prior.  """

    def __init__(self, likelihood, prior):
        """
        Initialise a conjugate prior exponential family.
        @arg likelihood: The exponential family whose parameters we have a
          conjugate prior for.
        @arg prior: The exponential family that is a conjugate prior over
          the likelihood's parameters.
        """
        self.likelihood = likelihood
        "The exponential family whose parameters we have a conjugate prior for"

        self.prior = prior
        """The exponential family that is a conjugate prior over the
        likelihood's parameters"""

        self.likelihood_dimension = self.likelihood.dimension
        """The dimension of the natural parameters and sufficient statistics
        of the likelihood."""

        self.strength_dimension = self.prior.dimension - \
            self.likelihood_dimension
        "The dimension of the strength part of the prior's natural parameters."

        assert self.strength_dimension == \
            self.likelihood.normalisation_dimension

    def add_observations_to_prior(self, T, tau, n=1):
        """
        Update the prior, tau, with the sum of the n observations, T.
        """
        d = self.strength_dimension
        _lambda = tau.copy()
        _lambda[:d] += n
        _lambda[d:] += T
        return _lambda

    def log_conjugate_predictive(self, T, tau, n=1):
        """
        @arg T: the data drawn from the likelihood.
        @arg tau: the natural parameter for this conjugate prior distribution
        @arg n: the number of data drawn from the likelihood.
        log of the predictive pdf for data, T, from the likelihood,
        given the prior's parameters, tau.
        """
        assert self.likelihood._check_shape(T)
        _lambda = self.add_observations_to_prior(T, tau, n)
        return self.prior.A(_lambda) - self.prior.A(tau) + self.likelihood.h(T)


class GaussianConjugatePrior(ConjugatePrior):

    """
    Univariate Normal distribution with a Normal-Gamma conjugate prior.
    """

    def __init__(self, p=2):
        "Initialise the Normal-Gamma - Univariate Normal conjugate prior pair"
        ConjugatePrior.__init__(
            self, GaussianExpFamily(), NormalGammaExpFamily(p))

    def exp_likelihood_log_normalisation_factor(self, tau):
        """
        @arg tau: the parameters of the prior in standard form
        the expectation of the log normalisation factor
        of the likelihood given the prior's parameters
        """
        alpha, beta, mu_0, _lambda = self.prior.theta(tau)
        return -.5 * (
            (digamma(alpha) - log(beta))
            - (_lambda + alpha / beta * mu_0 ** 2)
        )


class MvnConjugatePrior(ConjugatePrior):

    """
    Multivariate Normal distribution with a Normal-Wishart conjugate prior.
    """

    def __init__(self, p=2):
        """Initialise the Normal-Wishart - Multivariate Normal conjugate prior
        pair"""
        ConjugatePrior.__init__(
            self, MvnExpFamily(p), NormalWishartExpFamily(p))

    def exp_likelihood_log_normalisation_factor(self, tau):
        """
        The expectation of the log normalisation factor of the
        likelihood given the prior's parameters, tau in standard form.
        """
        nu, S, kappa_0, mu_0 = self.prior.theta(tau)
        return (
            self.prior.p / kappa_0
            + nu * dot(mu_0, dot(S, mu_0))
            + self.prior.p * log(pi)
            - log(det(S))
            - sum(digamma((nu - i) / 2.) for i in xrange(self.prior.p))
        ) / 2.


class MultinomialConjugatePrior(ConjugatePrior):

    """
    Multinomial distribution with a Dirichlet conjugate prior.
    """

    def __init__(self, k=2):
        "Initialise the conjugate prior pair"
        ConjugatePrior.__init__(
            self, DiscreteExpFamily(k), DirichletExpFamily(k))

    def exp_likelihood_log_normalisation_factor(self, tau):
        """
        @arg tau: the parameters of the prior in standard form
        the expectation of the log normalisation factor of the
          likelihood given the prior's parameters
        """
        return 0.
