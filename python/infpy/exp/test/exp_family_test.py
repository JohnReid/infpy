#
# Copyright John Reid 2007
#


import unittest, infpy.exp, scipy.stats, numpy, math
from numpy import log, outer, dot, asarray, zeros, empty, identity
from numpy.linalg import det, inv
from scipy.special import digamma

_epsilon = math.sqrt(numpy.finfo(float).eps)

def check_gradient(f, df, x0, tol=2e-2, text=''):
    "Checks the gradient of f matches the gradient calculated by df at x0"
    from scipy.optimize import approx_fprime
    approx = approx_fprime(x0, f, _epsilon)
    calculated = df(x0)
    diff = math.sqrt(sum((calculated-approx)**2))
    assert diff < tol, \
            "%s: Gradients don't match\nx0: %s\ndistance: %f\ncalculated: %s\napprox: %s\ndiff: %s" % (
             text, x0, diff, calculated, approx, calculated-approx
            )

def check_close(a, b, tol=1e-8):
    #special case for -1.#INF for discrete distribution
    try:
        if (a == b).all():
            return True
    except:
        pass

    diff = numpy.array(a) - numpy.array(b)
    diff_size = numpy.dot(diff, diff)
    return (diff_size < tol).all()

def check_is_close(u, v, tol=1e-4, strong_test=True, dont_test_zeros=False):
    """
    See U{http://www.boost.org/libs/test/doc/components/test_tools/floating_point_comparison.html} for
    partial inspiration.
    """
    u = numpy.asarray(u)
    v = numpy.asarray(v)
    assert u.shape == v.shape, 'u and v are not the same shape: %s != %s' % (str(u.shape), str(v.shape))

    if (u == v).all():
        return True

    abs_u = numpy.fabs(u)
    abs_v = numpy.fabs(v)
    diff = numpy.fabs(u - v)
    diff_u = diff / abs_u
    diff_v = diff / abs_v
    check_u = (diff_u <= tol)
    check_v = (diff_v <= tol)

    if dont_test_zeros:
        # if value is 0.0 allow any difference
        check_u += (0.0 == abs_u)
        check_v += (0.0 == abs_v)

    #print check_u, check_v

    if strong_test:
        return (check_u * check_v).all()
    else:
        return (check_u + check_v).all()

class CheckExpFamily(object):
    "Checks various methods/properties of an exponential family."

    def __init__(self, family):
        self.family = family
        "The exponential family being checked."

    def check_attributes(self):
        "Checks the class has the correct attributes."
        assert hasattr(self.family, 'T')
        assert hasattr(self.family, 'eta')
        assert hasattr(self.family, 'A')
        assert hasattr(self.family, 'h')
        assert hasattr(self.family, 'theta')
        assert hasattr(self.family, 'x')
        assert hasattr(self.family, 'dimension')
        assert hasattr(self.family, '_p_truth')
        assert hasattr(self.family, '_typical_xs')
        assert hasattr(self.family, '_typical_thetas')

    def check_xs(self):
        "Checks the family converts between x and T ok."
        for x in self.family._typical_xs:
            T = self.family.T(x)
            if tuple == type(x):
                test_x = self.family.x(self.family.T(x))
                for x_i, test_x_i in zip(x, test_x):
                    assert check_is_close(x_i, test_x_i), str(x)
            else:
                assert check_is_close(x, self.family.x(self.family.T(x))), str(x)
            assert len(T) == self.family.dimension
            assert check_is_close(T, self.family.T(self.family.x(T))), str(T)

    def check_dimensions(self):
        "Checks the family has the correct length for A(), T() and eta()."
        for x in self.family._typical_xs:
            T = self.family.T(x)
            T_shape = T.shape
            expected_T_shape = (self.family.dimension,)
            assert T_shape == expected_T_shape, "%s: T: incorrect shape/dimensions: %s != %s" % (
             self.family,
             str(T_shape),
             str(expected_T_shape)
            )
        for theta in self.family._typical_thetas:
            eta = self.family.eta(theta)
            eta_shape = eta.shape
            expected_eta_shape = (self.family.dimension,)
            assert eta_shape == expected_eta_shape, "%s: eta: incorrect shape/dimensions: %s != %s" % (
                self.family,
                str(eta_shape),
                str(expected_eta_shape)
            )
            A_shape = self.family.A_vec(eta).shape
            expected_A_shape = (self.family.normalisation_dimension,)
            assert A_shape == expected_A_shape, "%s: A: Incorrect normalisation dimension: %s != %s" % (
                self.family,
                str(A_shape),
                str(expected_A_shape)
            )

    def check_thetas(self):
        "Checks the family converts between eta and theta ok."
        for theta in self.family._typical_thetas:
            eta = self.family.eta(theta)
            assert len(eta) == self.family.dimension
            assert check_is_close(eta, self.family.eta(self.family.theta(eta)), strong_test=False, dont_test_zeros=True), str(eta)

    def check_against_truth(self):
        "Checks the family pdf agrees with the independently calculated pdf."
        for x in self.family._typical_xs:
            for theta in self.family._typical_thetas:
                calculated = self.family.p_x(x, theta)
                truth = self.family._p_truth(x, theta)
                #print x, theta, calculated, truth
                assert check_is_close(calculated, truth, strong_test=False, dont_test_zeros=True), \
                        '%s: True pdf does not match calculated pdf\nx:%s\ntheta:%s\ncalculated:%f\ntruth:%f' % (
                         str(self.family.__class__), str(x), str(theta), calculated, truth
                        )

    def check_entropy_truth(self):
        "Checks the family entropy agrees with the independently calculated entropy."
        for theta in self.family._typical_thetas:
            eta = self.family.eta(theta)
            calculated = self.family.entropy(eta)
            truth = self.family._entropy_truth(theta)
            assert check_is_close(calculated, truth), \
                    '%s: True entropy does not match calculated entropy\ntheta:%s; eta:%s; calculated:%f; truth:%f' % (
                     str(self.family.__class__), str(theta), str(eta), calculated, truth
                    )

    def check_exp_T(self, size=10000):
        "Checks the expectation of sufficient statistics."
        for theta in self.family._typical_thetas:
            eta = self.family.eta(theta)

            # check that the derivative of the log normalisation factor is the expectation of the sufficient statistic
            #check_gradient(
            # self.family.A,
            # self.family.exp_T,
            # eta,
            # text=str(self.family.__class__)
            #)

            # check the expectation of the sufficient statistic by sampling
            exp_T = self.family.sample(eta, size).sum(axis=0)/size
            assert check_is_close(exp_T, self.family.exp_T(eta), tol=.4, strong_test=False, dont_test_zeros=True), \
                    '%s: expected T != sampled T\ntheta:%s\neta:%s\nsampled:%s\ncalculated:%s' % (
                     str(self.family.__class__),
                     str(theta),
                     str(eta),
                     str(exp_T),
                     str(self.family.exp_T(eta))
                    )

    def _sample_KL(self, eta1, eta2, samples):
        return sum((self.family.log_p_T(sample, eta1) - self.family.log_p_T(sample, eta2)) for sample in samples) / len(samples)

    def _check_KL(self, eta1, eta2, samples):
        calculated_KL = self.family.KL(eta1, eta2)
        sampled_KL = self._sample_KL(eta1, eta2, samples)
        return calculated_KL, sampled_KL

    def check_KL(self, size=1000):
        "Checks the KL divergence by sampling."
        for theta1 in self.family._typical_thetas:
            eta1 = self.family.eta(theta1)
            samples = self.family.sample(eta1, size)

            for theta2 in self.family._typical_thetas:
                eta2 = self.family.eta(theta2)
                calculated_KL, sampled_KL = self._check_KL(eta1, eta2, samples)

                if (eta1 == eta2).all():
                    # KL should be 0 if eta1 == eta2
                    assert check_is_close(
                     1.0,
                     calculated_KL + 1.,
                     tol=1e-10,
                     strong_test=True,
                     dont_test_zeros=False
                    ), '%s: KL should be 0.0\ntheta1:%s\ntheta2:%s\ncalculated:%s' % (
                            str(self.family.__class__),
                            str(theta1),
                            str(theta2),
                            str(calculated_KL)
                    )

                # check the KL by sampling
                assert check_is_close(
                 sampled_KL,
                 calculated_KL,
                 tol=.3,
                 strong_test=False,
                 dont_test_zeros=True
                ), '%s: expected KL != sampled KL\ntheta1:%s\ntheta2:%s\nsampled:%s\ncalculated:%s' % (
                        str(self.family.__class__),
                        str(theta1),
                        str(theta2),
                        str(sampled_KL),
                        str(calculated_KL)
                )

        return sampled_KL - calculated_KL

    def check_vectorisation(self):
        "Checks the family handles multiple xs and etas at once."
        if self.family.vectorisable:
            xs = asarray(self.family._typical_xs)
            thetas = self.family._typical_thetas
            etas = empty((len(thetas), self.family.dimension))
            for k, theta in enumerate(thetas):
                etas[k] = self.family.eta(theta)
            
            Ts = self.family.T(xs)
            for x, T in zip(xs, Ts):
                self.family._check_shape(x)
                self.family._check_shape(T)
                assert (self.family.T(x) == T).all()
                
            _xs = self.family.x(Ts)
            for x, _x in zip(xs, _xs):
                check_is_close(x, _x)
            
            _thetas = self.family.theta(etas)
            for theta, _theta in zip(thetas, _thetas):
                check_is_close(theta, _theta)
    
            if hasattr(self.family, 'exp_T'):
                exp_Ts = self.family.exp_T(etas)
                for eta, exp_T in zip(etas, exp_Ts):
                    assert (self.family.exp_T(eta) == exp_T).all()
    
            As = self.family.A_vec(etas)
            for A, eta in zip(As, etas):
                check_is_close(A, self.family.A_vec(eta))

    def check_all(self):
        self.check_attributes()
        self.check_xs()
        self.check_thetas()
        self.check_dimensions()
        self.check_against_truth()
        self.check_vectorisation()
        if hasattr(self.family, 'exp_T'):
            self.check_exp_T()
        if hasattr(self.family, 'KL'):
            self.check_KL()
        if hasattr(self.family, '_entropy_truth'):
            self.check_entropy_truth()


class GeneralisedExpFamilyTests(unittest.TestCase):
    """
    Test case for all exponential families with default __init__ arguments.
    """

    def _test_subclasses(self, cls):
        "Test all the subclasses of cls."
        for subclass in cls.__subclasses__():
            #print subclass
            CheckExpFamily(subclass()).check_all()
            self._test_subclasses(subclass)

    def test(self):
        "Test all the subclasses of infpy.exp.ExponentialFamily."
        self._test_subclasses(infpy.exp.ExponentialFamily)



class CheckConjugatePrior(object):
    """
    Checks various methods/properties of a conjugate prior.
    """

    def __init__(self, conj_prior):
        self.conj_prior = conj_prior
        "The conjugate prior being checked."

    def check_conjugacy(self):
        """
        Test that the conjugate prior is set up correctly.
        """
        for theta in self.conj_prior.likelihood._typical_thetas:
            eta = self.conj_prior.likelihood.eta(theta)
            A = self.conj_prior.likelihood.A_vec(eta)
            _lambda = self.conj_prior.prior.T(theta)
            _lambda_2 = _lambda[:self.conj_prior.strength_dimension]
            assert check_is_close(
                _lambda_2,
                -A,
                tol=1.-5,
                strong_test=True,
                dont_test_zeros=False
            ), \
                'lambda_2 != -A\neta=%s\nlambda=%s\nlambda_2=%s\nA=%s' % (
                 str(eta),
                 str(_lambda),
                 str(_lambda_2),
                 str(A)
            )

    def check_log_normalisation_factor_expectation(self, sample_size=1000):
        """
        Test that we get close to the same expectation of the mvn normalisation factor from sampling from
        a normal wishart that we do from calculation of our formula.
        """
        for theta in self.conj_prior.prior._typical_thetas:
            eta = self.conj_prior.prior.eta(theta)
            calculated_lnf = self.conj_prior.exp_likelihood_log_normalisation_factor(eta)
            samples = self.conj_prior.prior.sample(eta, size=sample_size)
            sampled_lnf = 0.
            for sample in samples:
                sampled_lnf += self.conj_prior.likelihood.A(sample[self.conj_prior.strength_dimension:])
            sampled_lnf /= sample_size
            assert check_is_close(
             calculated_lnf,
             sampled_lnf,
             tol=.1,
             strong_test=False,
             dont_test_zeros=True
            ), \
                    '<log A(eta)>: expected != sampled\neta:%s\ncalculated:%s\nsampled:%s' % (
                     str(eta),
                     str(calculated_lnf),
                     str(sampled_lnf)
            )


    def check_all(self):
        """
        Run all checks on this conjugate prior.
        """
        self.check_conjugacy()
        self.check_log_normalisation_factor_expectation()




class GeneralisedConjPriorTests(unittest.TestCase):
    """
    Test case for all conjugate priors.
    """

    def _test_subclasses(self, cls):
        "Test all the subclasses of cls."
        for subclass in cls.__subclasses__():
            #print subclass
            CheckConjugatePrior(subclass()).check_all()
            self._test_subclasses(subclass)

    def test(self):
        "Test all the subclasses of infpy.exp.ConjugatePrior."
        self._test_subclasses(infpy.exp.ConjugatePrior)



class MvnConjPriorTests(unittest.TestCase):
    """
    Tests conjugate prior of MVN distribution.
    """

    def test_expectations_by_sampling(self, sample_size=1000):
        """
        Test that we get close to the same expectation of the mvn normalisation factor from sampling from
        a normal wishart that we do from calculation of our formula.
        """
        from infpy.exp import WishartExpFamily, MvnExpFamily

        k = 2
        wishart = WishartExpFamily(k)
        mvn = MvnExpFamily(k)

        for kappa_0 in [ 1., 2. ]:
            for mu_0 in [ [0., 0.], [5., -3.] ]:
                for nu, S in WishartExpFamily._typical_thetas:

                    calculated_mu_W_mu = k/kappa_0 + nu * dot(mu_0, dot(S, mu_0))
                    calculated_log_W = sum(digamma((nu-i)/2.) for i in xrange(k)) + log(det(S)) + k*log(2.)

                    sampled_W = wishart.sample(eta=wishart.eta((nu,S)), size=sample_size)
                    sampled_mu_W_mu = 0.0
                    sampled_log_W = 0.0
                    for T_W in sampled_W:
                        W = wishart.x(T_W)
                        #print log(det(W))
                        sampled_log_W += log(det(W))

                        precision = kappa_0 * W
                        mu = mvn.x(mvn.sample(eta=mvn.eta((mu_0, precision)), size=1)[0])
                        sampled_mu_W_mu += dot(mu, dot(W, mu))
                        #print W, mu
                    sampled_mu_W_mu /= sample_size
                    sampled_log_W /= sample_size

                    assert check_is_close(
                                             calculated_mu_W_mu,
                                             sampled_mu_W_mu,
                                             tol=.1,
                                             strong_test=False,
                                             dont_test_zeros=True
                                            ), \
                                                    'log|W|: expected != sampled\nkappa_0:%s\nmu_0:%s\nnu:%s\nS:%s\nsampled:%s\ncalculated:%s' % (
                                                     str(kappa_0),
                                                     str(mu_0),
                                                     str(nu),
                                                     str(S),
                                                     str(sampled_mu_W_mu),
                                                     str(calculated_mu_W_mu)
                                            )

                    assert check_is_close(
                                             calculated_log_W,
                                             sampled_log_W,
                                             tol=.3,
                                             strong_test=False,
                                             dont_test_zeros=True
                                            ), \
                                                    'log|W|: expected != sampled\nkappa_0:%s\nmu_0:%s\nnu:%s\nS:%s\nsampled:%s\ncalculated:%s' % (
                                                     str(kappa_0),
                                                     str(mu_0),
                                                     str(nu),
                                                     str(S),
                                                     str(sampled_log_W),
                                                     str(calculated_log_W)
                                            )

class HighDMvnTest(unittest.TestCase):
    """
    Test case for high dimensional multivariate normal.
    """

    def setUp(self):
                    # Create a sequence of exponential families of high dimension
        self.families = [
         infpy.exp.MvnExpFamily(k=dim)
         for dim in xrange(2,10) # check up to 10-D
        ]

        # give each family a typical set of xs and thetas
        for family in self.families:
            family._typical_xs = [
             zeros((family.k,)) + 1.1
            ]
            family._typical_thetas = [
             (
              zeros((family.k,)) + 1.01,
              identity(family.k)
             )
            ]

    def test_exp_family_std_methods(self):
        for family in self.families:
            # run the tests
            CheckExpFamily(family).check_all()

    def test_expectations_by_sampling(self, size=1000):
        "Test some expected values by sampling from the distribution."
        for family in self.families:
            for theta in family._typical_thetas:
                eta = family.eta(theta)
                sample = family.sample(eta, size)
                xs = [family.x(T) for T in sample]
                mu, W = theta

                # check expected value of x
                sampled_x = sum(xs)/size
                exp_x = mu
                assert check_is_close(exp_x, sampled_x, tol=.3, strong_test=False, dont_test_zeros=True), \
                    'x: expected != sampled\ntheta:%s\neta:%s\nsampled:%s\ncalculated:%s' % (
                     str(theta),
                     str(eta),
                     str(sampled_x),
                     str(exp_x)
                )

                # check expected value of x.x'
                sampled_x2 = sum(outer(x, x) for x in xs)/size
                exp_x2 = outer(mu, mu) + inv(W)
                assert check_is_close(exp_x2, sampled_x2, tol=.2, strong_test=False, dont_test_zeros=True), \
                    'x.x\': expected != sampled\ntheta:%s\neta:%s\nsampled:%s\ncalculated:%s' % (
                     str(theta),
                     str(eta),
                     str(sampled_x2),
                     str(exp_x2)
                )


if __name__ == "__main__":
    # GeneralisedConjPriorTests('test').debug()
    # GeneralisedExpFamilyTests('test').debug()
    #CheckExpFamily(infpy.exp.GaussianExpFamily()).check_all()
    #CheckExpFamily(infpy.exp.NormalGammaExpFamily()).check_all()
    #CheckConjugatePrior(infpy.exp.GaussianConjugatePrior()).check_all()
    #CheckExpFamily(infpy.exp.DirichletExpFamily()).check_all()
    #HighDMvnTest('test_exp_family_std_methods').debug()
    #GeneralisedExpFamilyTests('test').debug()
    unittest.main()
