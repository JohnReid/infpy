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
            self._T = t
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


class CRPMixture(object):

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
        >>> mixture = crp.CRPMixture(crpalpha, F, T, tau)
        >>> K = []
        >>> numsamples = 100
        >>> for i in xrange(numsamples):
        ...     mixture.sample()
        ...     K.append(len(npy.unique(mixture.crp.z)))

    """

    def __init__(self, alpha, F, x, tau):
        self.crp = CRP(alpha)
        self.F = F
        self.x = x
        self.tau = tau
        self.psi = []  # Posteriors for cluster parameters.
        for n in xrange(self.N):
            self.sampleandsit(n)

    @property
    def N(self):
        "Number of data."
        return len(self.x)

    def tablelikelihoodfn(self, n, t):
        "The likelihood of the n'th datum under the t'th table's parameter."
        # Is it a likelihood for a new table?
        if -1 == t:
            tau = self.tau
        else:
            tau = self.psi[t]
        _logger.debug('n=%d; T=%s, tau=%s', n, self.x[n], tau)
        return npy.exp(self.F.log_conjugate_predictive(self.x[n], tau))

    def sample(self):
        "Make one datum stand up and sit down."
        n = npy.random.choice(self.N)
        t = self.crp.standup(n)
        self.standup(n, t)
        self.sampleandsit(n)

    def standup(self, n, t):
        self.psi[t] = self.F.add_observations_to_prior(-self.x[n], self.psi[t], n=-1)

    def sitdown(self, n, t):
        if t >= len(self.psi):
            assert t == len(self.psi)  # Can only have added one table
            self.psi.append(self.tau.copy())
        self.psi[t] = self.F.add_observations_to_prior(self.x[n], self.psi[t], n=1)

    def sampleandsit(self, n):
        t = self.crp.sampleandsit(n, self.tablelikelihoodfn)
        self.sitdown(n, t)
        return t


class DP(object):

    """A `Dirichlet process <http://en.wikipedia.org/wiki/Dirichlet_process>`_.
    """

    def __init__(self, H, alpha):
        self.crp = CRP(alpha)
        self.H = H

    def sample(self, n):
        t = self.crp.sampletable(n, self.likelihoodfn)
        if t >= len(self.atoms):
            assert t == len(self.atoms)  # Can only have added one table
        # Sample a new atom if it is a new table
        self.atoms.append(self.H.sample())
        return t

    def likelihoodfn(self, t):
        return 1.


class CCModel(object):

    def __init__(self, x, family, H, beta):
        self.maxdim = max(f.dimension for f in family)
        self.family = family
        self.H = H
        self.x = x
        self.beta = beta
        # Use -1 as indicator that table is not assigned
        self.k = list()
        self.z = - npy.ones(self.N)
        self._initialisepsiprior()
        self._initialisestate()

    def _initialisepsiprior(self):
        self.psiprior = npy.zeros((self.J, self.maxdim))
        for j in xrange(self.J):
            self.psiprior[j, :self.family[j].dimension] = self.H[j]
        self.psi = [self.psiprior.copy()]

    def _initialisestate(self):
        "Initialise the state by sampling z."
        for n in xrange(self.N):
            self._samplez(n)

    def _getJ(self):
        return len(self.H)

    J = property(_getJ, "The number of contexts.")

    def _getN(self):
        return len(self.x)

    N = property(_getN, "The number of data.")

    def _getT(self):
        return len(self.k)

    T = property(_getT, "The number of tables in G.")

    def _getm(self):
        m = npy.zeros(self.T, dtype=int)
        for zn in self.z:
            m[zn] += 1
        return m

    m = property(_getm, "The number of customers at each table in G.")

    def _standup(self, n):
        "Stand the n'th datum up from the table it is sitting at."
        t = self.z[n]
        if -1 != t:
            self.z[n] = -1
            self._sitattable(-self.x[n], t)

    def _sitdown(self, n, t):
        "Sit the n'th datum down at the t'th table."
        self.z[n] = t
        self._sitattable(self.x[n], t)

    def _sitattable(self, xn, t):
        "Sit the datum at table t. This updates self.psi."
        if t < 0 or t >= len(self.T):
            raise ValueError("t=%d does not index a table" % t)
        k = self.k[t]
        for j, kj in enumerate(k):
            self.psi[kj, j] += xn[j]

    def _createnewtable(self):
        "Sample a new table - initialise psi with the prior."
        self.psi.append(self.psiprior.copy())

    def _samplez(self, n):
        "Sample a table assignment for n'th datum."
        xn = self.x[n]
        t = self.z[n]
        # If they are already seated, we must make them stand up
        self._standup(n)
        # Calculate the probabilities to sit at any existing table
        # or create a new one
        p = npy.empty(self.T + 1)
        m = self.m
        emptytable = None  # Check for empty tables we can reuse
        for t in range(self.T):
            if 0 == m[t]:
                emptytable = t
                p[t] = 0
            else:
                p[t] = m[t] * self.tablelikelihood(t, xn)
        p[self.T] = self.beta * self.priorlikelihood(xn)
        # Choose a table to sit at
        t = npy.random.choice(self.T + 1, p)
        # If we chose a new table, we must create it unless we can
        # reuse an old empty one
        if t == self.T:
            if emptytable is None:
                self._createnewtable()
            else:
                t = emptytable
        # Sit down
        del m  # m no longer valid once we sit down
        self.sitdown(n, t)

    def priorlikelihood(self, xn):
        "The likelihood of xn given the priors for each context."
        return npy.exp(sum(
            family.log_p_T(xnj, eta)
            for family, eta, xnj
            in zip(self.family, self.psiprior, xn)
        ))

    def tablelikelihood(self, t, xn):
        "The likelihood of xn given the parameters of table t."
        return npy.exp(sum(
            family.log_p_T(xnj, self.psi[k][j])
            for family, k, xnj, j
            in zip(self.family, self.k[t], xn, xrange(self.J))
        ))


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

    # import pandas as pd
    # tcga = pd.read_csv("../data/TCGA.csv.gz", compression="gzip")
    import infpy.exp as exp
    families = [
        exp.GaussianExpFamily(dimension=1),
        exp.GaussianExpFamily(dimension=1),
    ]
    x = npy.array([
        [1.0, 1],
        [1.0, 1],
        [1.0, 1],
        [0.0, 2],
        [1.0, 2],
        [0.0, 2],
        [0.0, 2],
    ])
    model = CCModel(x)
