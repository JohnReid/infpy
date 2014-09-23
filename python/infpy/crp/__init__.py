#
# Copyright John Reid 2014
#

import logging
logging.basicConfig(level=logging.DEBUG)
_logger = logging.getLogger(__name__)

import numpy as npy


class CRP(object):

    """ A `Chinese restaurant process
    <http://en.wikipedia.org/wiki/Chinese_restaurant_process>`_.

    """

    def __init__(self, alpha):
        self.alpha = alpha
        self.z = []  # Assignments of data to tables
        self.m = []  # Number of customers (data) at each table

    @property
    def T(self):
        "The number of tables."
        return len(self.m)

    @property
    def N(self):
        "The number of data."
        return len(self.z)

    def __repr__(self):
        "String representation."
        return "CRP with {} tables for {} data".format(self.T, self.N)

    def issitting(self, n):
        "Is the n'th datum sitting down."
        return -1 != self.z[n]

    def standup(self, n):
        "Stand the n'th datum up from the table it is sitting at."
        t = self.z[n]
        if -1 == t:
            raise ValueError('Cannot stand up, not sitting down.')
        self.m[t] -= 1
        self.z[n] = -1
        return t

    def sitdown(self, n, t):
        "Sit the n'th datum down at the t'th table."
        # Extend table assignments list if necessary
        if len(self.z) <= n:
            self.z.extend([-1] * (n + 1 - len(self.z)))
        if -1 != self.z[n]:
            raise ValueError('Cannot sit down, already sitting down.')
        self.z[n] = t
        # Update the number of tables if it has increased.
        if t >= self.T:
            self.m.append(1)
        else:
            self.m[t] += 1

    def sampletable(self, n, tablelikelihoodfn):
        """Choose a table (new or old) given the likelihoods defined for each table
        by the tablelikelihoodfn."""
        # Calculate the probabilities to sit at any existing table
        # or create a new one
        T = self.T
        p = npy.empty(T + 1)
        m = self.m
        emptytable = None  # Check for empty tables we can reuse
        for t in range(T):
            if 0 == m[t]:
                emptytable = t
                p[t] = 0
            else:
                p[t] = m[t] * tablelikelihoodfn(n, t)
        p[T] = self.alpha * tablelikelihoodfn(n, -1)
        # Choose a table to sit at
        t = npy.random.choice(range(T + 1), p=p/p.sum())
        _logger.debug('sampletable(): choice = %d; weights = %s', t, p)
        # If we chose a new table, we must create it unless we can
        # reuse an old empty one
        if t == T and emptytable is not None:
            t = emptytable
        return t

    def sampleandsit(self, n, tablelikelihoodfn):
        "Sample a table for the n'th datum and sit at it."
        t = self.sampletable(n, tablelikelihoodfn)
        self.sitdown(n, t)
        return t

    def flip(self, n, tablelikelihoodfn):
        "Change the status of datum n (sitting or standing)."
        if self.issitting(n):
            self.standup(n)
        else:
            self.sampleandsit(n, tablelikelihoodfn)


class DPExpFam(object):

    """A `Dirichlet process <http://en.wikipedia.org/wiki/Dirichlet_process>`_
    that samples atoms from an exponential family conjugate prior.
    """

    def __init__(self, alpha, F, x, tau):
        self.crp = CRP(alpha)
        self.F = F
        self.x = x
        self.tau = tau
        self.psi = []  # Posteriors for cluster parameters.

    def tablelikelihood(self, n, t):
        """The likelihood of the n'th datum given the parameter
        of the t'th table."""
        return npy.exp(self.tableloglikelihood(n, t))

    def tableloglikelihood(self, n, t):
        """The log likelihood of the n'th datum given the parameter
        of the t'th table."""
        if -1 == t:
            psi = self.tau
        else:
            psi = self.psi[t]
        logp = self.F.log_conjugate_predictive(self.x[n], psi)
        _logger.debug('n=%d; t=%d, p=%.2e; T=%s, tau=%s',
                      n, t, npy.exp(logp), self.x[n], psi)
        return logp

    @property
    def N(self):
        "Number of data."
        return len(self.x)

    def standup(self, n, t):
        "Stand the n'th observation up from the t'th table."
        self.psi[t] = self.F.add_observations_to_prior(-self.x[n],
                                                       self.psi[t],
                                                       n=-1)

    def sitdown(self, n, t):
        "Sit the n'th observation down at the t'th table."
        if t >= len(self.psi):
            assert t == len(self.psi)  # Can only have added one table
            self.psi.append(self.tau.copy())
            assert len(self.psi) == self.crp.T
        self.psi[t] = self.F.add_observations_to_prior(self.x[n],
                                                       self.psi[t],
                                                       n=1)

    def tablelikelihoodfn(self, n, t):
        "The likelihood of the n'th datum under the t'th table's parameter."
        # Is it a likelihood for a new table?
        if -1 == t:
            psi = self.tau
        else:
            psi = self.psi[t]
        _logger.debug('n=%d; T=%s, tau=%s', n, self.x[n], psi)
        return npy.exp(self.F.log_conjugate_predictive(self.x[n], psi))


class DPMixture(DPExpFam):

    """A mixture of distributions from an exponential family
    implemented on top of a Chinese restaurant process.

.. doctest::

        >>> import infpy.crp as crp
        >>> import infpy.exp as exp
        >>> import numpy as npy
        >>> F = exp.GaussianConjugatePrior()
        >>> x = npy.array([0, 0, 0, 1, 1, 1])
        >>> T = F.likelihood.T(x)
        >>> crpalpha = 1.
        >>> prioralpha = 100.
        >>> priorbeta = 1.
        >>> priormu = 0.5
        >>> priorlambda = 100.
        >>> tau = F.prior.eta((prioralpha, priorbeta, priormu, priorlambda))
        >>> mixture = crp.DPMixture(crpalpha, F, T, tau)
        >>> K = []
        >>> numsamples = 100
        >>> for i in xrange(numsamples):
        ...     mixture.sample()
        ...     K.append(len(npy.unique(mixture.crp.z)))

    """

    def __init__(self, alpha, F, x, tau):
        super(DPMixture, self).__init__(alpha, F, x, tau)
        for n in xrange(self.N):
            self.sampleandsit(n)

    def sample(self):
        "Make one datum stand up and sit down."
        n = npy.random.choice(self.N)
        t = self.crp.standup(n)
        self.standup(n, t)
        self.sampleandsit(n)

    def sampleandsit(self, n):
        t = self.crp.sampleandsit(n, self.tablelikelihood)
        self.sitdown(n, t)
        return t


class ContextClusterMixture(object):

    r"""
    A context clustering mixture model. Each datum takes values in a set
    of contexts.

    - family = (H, F): A list of exponential family conjugate
      prior/likelihood pairs (one for each context).
    - x: The data, indexed by context first then datum.
    - tau: A list of priors for each context.
    - alpha: The parameter for the context level DPs.
    - beta: The parameter for the overall DP.

    If we have :math:`N` data each with values in :math:`J` contexts,
    then we have the following model:

    .. math::

        G_j | H_j, \alpha \sim \textrm{DP}(H_j, \alpha)

        G_0 | G_1, \dots, G_J, \beta \sim \textrm{DP}(G_1
            \otimes \dots \otimes G_J, \beta)

        \theta_n | G_0 \sim G_0

        x_{n,j} | \theta_{n,j} \sim F_j(\theta_{n,j})

    """

    def __init__(self, family, x, tau, alpha, beta):
        lengths = map(len, x)
        if not lengths:
            raise ValueError('No contexts')
        if max(lengths) != min(lengths):
            raise ValueError('Different number of data in different contexts.')
        self.N = lengths[0]
        "Number of data."
        self.contexts = [
            self.Context(alpha, _family, _x, _tau)
            for _family, _x, _tau
            in zip(family, x, tau)
        ]
        "The contexts."
        self.G0 = CRP(beta)
        self._initialisestate()

    def _initialisestate(self):
        "Initialise the state sitting every datum down."
        for n in xrange(self.N):
            self.sampleandsit(n)

    @property
    def J(self):
        "The number of contexts."
        return len(self.contexts)

    class Context(DPExpFam):

        """Holds the context specific data and parameters."""

        def __init__(self, alpha, F, x, tau):
            super(ContextClusterMixture.Context, self).__init__(alpha, F, x,
                                                                tau)

        def likelihoods(self, n):
            "The likelihood of the n'th datum for each table."
            lls = npy.empty(self.crp.T + 1)
            for t in xrange(self.crp.T):
                if self.crp.m > 0:
                    lls[t] = self.tableloglikelihood(n, t)
                else:
                    assert 0 == self.crp.m
                    lls[t] = -npy.inf
            lls[-1] = self.tableloglikelihood(n, -1)
            return npy.exp(lls)

        def newtablelikelihood(self, likelihoods):
            return (
                self.crp.alpha * likelihoods[-1]  # new table
                + npy.dot(self.crp.m, likelihoods[:-1])  # sum over existing
            ) / (sum(self.crp.m) + self.crp.alpha)

    class Likelihoods(object):

        """Calculates the likelihoods of each cluster context assignment."""

        def __init__(self, contexts, n):
            self.contexts = contexts
            self.n = n
            print 'Calculating likelihoods'
            self.likelihoods = map(lambda context: context.likelihoods(n),
                                   contexts)

        def tablelikelihood(self, n, t0):
            """The likelihood for datum n sitting at table t0 in
            the overall DP, G0."""
            assert n == self.n
            # Is it the likelihood for a new table in G0?
            if -1 == t0:
                # Yes - calculate the likelihood for each context as an
                # integral over the context's tables
                contextlikelihoods = (context.newtablelikelihood(likelihoods)
                                      for context, likelihoods
                                      in zip(self.contexts, self.likelihoods)
                                      )
            else:
                # No - calculate the likelihood for each context by
                # picking out the context's table that the table in G0
                # is associated with
                contextlikelihoods = (likelihoods[context.crp.z[t0]]
                                      for context, likelihoods
                                      in zip(self.contexts, self.likelihoods)
                                      )
            # Return the product over all contexts
            return reduce(float.__mul__, contextlikelihoods, 1.)

        def contexttablelikelihoodfn(self, j):
            """Return a function that calculates the likelihood for the
            n'th datum sitting at the tj'th table in context j.
            """
            def contexttablelikelihood(n, tj):
                return self.likelihoods[j][tj]
            return contexttablelikelihood

    def sampleandsit(self, n):
        likelihoods = self.Likelihoods(self.contexts, n)
        t0 = self.G0.sampleandsit(n, likelihoods.tablelikelihood)
        # Did we sit at a new table in G0?
        if 1 == self.G0.m[t0]:
            # Yes so choose a table in each context level DP
            for j, context in enumerate(self.contexts):
                context.crp.sampleandsit(
                    t0,
                    likelihoods.contexttablelikelihoodfn(j))
        # Update psi at context level
        for context in self.contexts:
            tj = context.crp.z[t0]
            context.sitdown(n, tj)


if '__main__' == __name__:

    # Test the Normal-Gamma / Gaussian conjugate prior
    import infpy.exp
    reload(infpy.exp)
    import infpy.exp as exp
    import pylab as pyl
    F = exp.GaussianConjugatePrior()
    prioralpha = 100.
    priorbeta = 1.
    priormu = 0.5
    priorlambda = 100.
    tau = F.prior.eta((prioralpha, priorbeta, priormu, priorlambda))
    numpriors = 100
    numsamples = 100
    means = []
    sds = []
    etas = F.prior.sample(tau, numpriors)
    for eta in etas:
        T = F.likelihood.sample(eta[F.strength_dimension:], size=numsamples)
        x = F.likelihood.x(T)
        means.append(npy.mean(x))
        sds.append(npy.std(x))
    pyl.figure()
    n, bins, patches = pyl.hist(means, 50, histtype='stepfilled')
    pyl.savefig('means.pdf')
    pyl.figure()
    n, bins, patches = pyl.hist(sds, 50, histtype='stepfilled')
    pyl.savefig('sds.pdf')

    # Test the CRP mixture model
    import infpy.exp
    reload(infpy.exp)
    import infpy.crp
    reload(infpy.crp)
    import infpy.crp as crp
    import infpy.exp as exp
    import numpy as npy
    F = exp.GaussianConjugatePrior()
    x = npy.array([0, 0, 0, 1, 1, 1])
    T = F.likelihood.T(x)
    crpalpha = 1.
    prioralpha = 100.
    priorbeta = 1.
    priormu = 0.5
    priorlambda = 100.
    tau = F.prior.eta((prioralpha, priorbeta, priormu, priorlambda))
    mixture = crp.CRPMixture(crpalpha, F, T, tau)
    K = []
    for i in xrange(1000):
        mixture.sample()
        K.append(len(npy.unique(mixture.crp.z)))

    # Test the context clustering model
    import logging
    logging.basicConfig(level=logging.DEBUG)
    import infpy.exp
    reload(infpy.exp)
    import infpy.exp as exp
    import numpy as npy
    F = exp.GaussianConjugatePrior()
    # Use two contexts with same exponential family
    x = [
        npy.array([0,  0,  0,  1,  1,  1]),
        npy.array([0,  0,  1,  1, -1, -1]),
        ]
    T = map(F.likelihood.T, x)  # Convert to exponential family suff. stats
    alpha = 1.
    beta = 1.
    prioralpha = 100.
    priorbeta = 1.
    priormu = 0.0
    priorlambda = 100.
    tau = F.prior.eta((prioralpha, priorbeta, priormu, priorlambda))
    import infpy.crp
    reload(infpy.crp)
    ccm = infpy.crp.ContextClusterMixture([F] * len(x), T, [tau] * len(x),
                                          alpha, beta)

    for i in xrange(1000):
        mixture.sample()
        K.append(len(npy.unique(mixture.crp.z)))
    # import pandas as pd
    # tcga = pd.read_csv("../data/TCGA.csv.gz", compression="gzip")
