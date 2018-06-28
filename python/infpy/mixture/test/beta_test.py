#
# Copyright John Reid 2010
#

"""
Code to test mixtures of beta distributions.
"""

import infpy.mixture.beta
reload(infpy.mixture.beta)
from infpy.mixture import beta
from infpy.convergence_test import check_LL_increased, LlConvergenceTest
import unittest
import logging
import numpy as np
from scipy.special import gammaln
from cookbook.cache_decorator import Memoize


@Memoize
def log_factorial(k):
    if 0 > k:
        raise ValueError('k must be > 0')
    elif int(k) != k:
        raise ValueError('k must be whole number')
    elif 0 == k:
        return 0.
    else:
        return np.log(k) + log_factorial(k - 1)


def pre_compute_log_factorial(N):
    for i in range(50, N, 50):
        log_factorial(i)


def binomial_coefficient(n, k):
    return np.exp(log_factorial(n) - log_factorial(k) - log_factorial(n - k))


class GreaterThanTest(unittest.TestCase):

    def test_greater_than(self):
        "Test greater than functionality."
        logging.info("Testing greater than functionality.")
        x = np.array([4, 3, 2, 1])
        gt = beta.greater_than(x)
        should_be = np.array([6, 3, 1, 0])
        assert (should_be == gt).all(
        ), 'greater_than(%s) = %s but it should be = %s' % (x, gt, should_be)

    def test_greater_than_2d(self):
        "Test greater than functionality."
        logging.info("Testing greater than functionality.")
        x = np.array([
            [4, 3, 2, 1],
            [5, 4, 3, 2],
        ])
        gt = beta.greater_than_2d(x)
        should_be = np.array([
            [6, 3, 1, 0],
            [9, 5, 2, 0],
        ])
        assert (should_be == gt).all(
        ), 'greater_than_2d(%s) = %s but it should be = %s' % (x, gt, should_be)

    def test_less_than_2d(self):
        "Test less than functionality."
        logging.info("Testing less than functionality.")
        x = np.array([
            [4, 3, 2, 1],
            [5, 4, 3, 2],
        ])
        lt = beta.less_than_2d(x)
        should_be = np.array([
            [0, 4, 7, 9],
            [0, 5, 9, 12],
        ])
        assert (should_be == lt).all(
        ), 'less_than_2d(%s) = %s but it should be = %s' % (x, lt, should_be)


class BetaTest(unittest.TestCase):

    def setUp(self):
        self.options = beta.get_default_options()
        self.options.max_iter = 200
        self.options.min_iter = 4

        self.K = 3

        # the exponential family we are dealing with
        self.exp_family = beta.DirichletExpFamily(k=2)

        np.random.seed(1)
        self.setUpData()

        #self.weights = np.ones(len(self.T))
        #self.weights[0] = .2
        self.weights = np.random.rand(len(self.T))

    def update_mixture(self, mixture, increase_tolerance=1e-6):
        # update mixture
        convergence_test = LlConvergenceTest(
            eps=1e-4, should_increase=True, use_absolute_difference=False)
        last_LL = None
        logging.info('Variational bound at start = %f',
                     mixture.variational_bound())
        for _i in range(self.options.max_iter):
            if None == last_LL:
                last_LL = mixture.variational_bound_piecewise()
                convergence_test(last_LL.sum())
            mixture.q_z.update(mixture)
            last_LL = check_LL_increased(last_LL, mixture.variational_bound_piecewise(
            ), tag="Update z", tolerance=increase_tolerance, raise_error=True)
            assert np.isfinite(last_LL).all(), str(last_LL)
            mixture._check_shapes()
            mixture.reorder_components()
            last_LL = check_LL_increased(last_LL, mixture.variational_bound_piecewise(
            ), tag="Reorder components", tolerance=increase_tolerance, raise_error=True)
            assert np.isfinite(last_LL).all(), str(last_LL)
            mixture._check_shapes()
            if hasattr(mixture, 'q_pi'):
                mixture.q_pi.update(mixture)
                assert .99 < mixture.q_pi.E().sum() < 1.01
                last_LL = check_LL_increased(last_LL, mixture.variational_bound_piecewise(
                ), tag="Update pi", tolerance=increase_tolerance, raise_error=True)
                assert np.isfinite(last_LL).all(), str(last_LL)
                mixture._check_shapes()
            mixture.q_eta.update(mixture)
            last_LL = check_LL_increased(last_LL, mixture.variational_bound_piecewise(
            ), tag="Update eta", tolerance=increase_tolerance, raise_error=False)
            assert np.isfinite(last_LL).all(), str(last_LL)
            mixture._check_shapes()
            logging.info('Iteration %d: variational bound = %f',
                         _i + 1, mixture.variational_bound())
            if _i + 1 >= self.options.min_iter and convergence_test(last_LL.sum()):
                logging.info('Variational bound has converged : stopping.')
                break

    def test_full_dist(self):
        "Test full distribution."
        logging.info('Testing full variational distributions : %s',
                     type(self).__name__)
        self.options.point_estimates = False
        first_seed = 1
        for seed in range(first_seed, first_seed + self.num_starts):
            logging.info('Seeding numpy.random with %d', seed)
            np.random.seed(seed)

            # create a mixture of exponential family distributions
            mixture = beta.ExpFamilyMixture(
                self.T,
                self.weights,
                self.K,
                self.exp_family,
                -np.ones(2),
                1.,
                options=self.options
            )
            # burn in 3 times to avoid nasty decreases in variational bound that happen at first few iterations
#            for _i in xrange(0):
#                mixture.update()
            self.update_mixture(mixture)

    def test_point_estimates(self):
        "Test point estimates for eta."

        logging.info('Testing with point estimates : %s', type(self).__name__)
        self.options.point_estimates = True
        first_seed = 1
        for seed in range(first_seed, first_seed + self.num_starts):
            logging.info('Seeding numpy.random with %d', seed)
            np.random.seed(seed)

            # create a mixture of exponential family distributions
            mixture = beta.ExpFamilyMixture(
                self.T,
                self.weights,
                self.K,
                self.exp_family,
                -np.ones(2),
                1.,
                options=self.options
            )
            self.update_mixture(mixture)

    def test_stick_breaking(self):
        "Test stick breaking prior for pi."

        logging.info(
            'Testing with stick-breaking prior for pi. : %s', type(self).__name__)
        self.options.point_estimates = True
        self.options.stick_breaking = True
        first_seed = 6
        for seed in range(first_seed, first_seed + self.num_starts):
            logging.info('Seeding numpy.random with %d', seed)
            np.random.seed(seed)

            # create a mixture of exponential family distributions
            mixture = beta.ExpFamilyMixture(
                self.T,
                self.weights,
                self.K,
                self.exp_family,
                -np.ones(2),
                1.,
                options=self.options
            )
            self.update_mixture(mixture)

    def test_integrate_out_dirichlet(self):
        "Test integrating out a Dirichlet distributed pi."

        logging.info(
            'Testing integrating out a Dirichlet distributed pi. : %s', type(self).__name__)
        self.options.point_estimates = True
        self.options.stick_breaking = False
        self.options.integrate_pi = True
        first_seed = 1
        for seed in range(first_seed, first_seed + self.num_starts):
            logging.info('Seeding numpy.random with %d', seed)
            np.random.seed(seed)

            # create a mixture of exponential family distributions
            mixture = beta.ExpFamilyMixture(
                self.T,
                self.weights,
                self.K,
                self.exp_family,
                -np.ones(2),
                1.,
                options=self.options
            )
#            for _ in xrange(2): # burn in a couple of times as we seem to have a problem with first few iterations...
#                mixture.update()
            self.update_mixture(mixture, increase_tolerance=1.)

    def test_integrate_out_stick(self):
        "Test integrating out a stick-breaking distributed pi."

        logging.info(
            'Testing integrating out a stick-breaking distributed pi. : %s', type(self).__name__)
        self.options.point_estimates = True
        self.options.stick_breaking = True
        self.options.integrate_pi = True
        first_seed = 1
        for seed in range(first_seed, first_seed + self.num_starts):
            logging.info('Seeding numpy.random with %d', seed)
            np.random.seed(seed)

            # create a mixture of exponential family distributions
            mixture = beta.ExpFamilyMixture(
                self.T,
                self.weights,
                self.K,
                self.exp_family,
                -np.ones(2),
                1.,
                options=self.options
            )
#            for _ in xrange(2): # burn in a couple of times as we seem to have a problem with first few iterations...
#                mixture.update()
            self.update_mixture(mixture, increase_tolerance=1.)


class BetaTestSinglePoint(BetaTest):
    def setUpData(self):
        # get sufficient statistics
        x = (.5,)
        X = np.empty((len(x), 2))
        X[:, 0] = x
        X[:, 1] = 1. - X[:, 0]
        self.T = self.exp_family.T(X)

        self.num_starts = 1


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


class BetaTestBigExample(BetaTest):
    def setUpData(self):
        # get sufficient statistics
        block_size = 300
        y = np.empty(3 * block_size)
        y[:block_size] = np.random.normal(loc=-10, scale=2., size=block_size)
        y[block_size:-
            block_size] = np.random.normal(loc=0, scale=4., size=block_size)
        y[-block_size:] = np.random.normal(loc=10, scale=2., size=block_size)
        x = sigmoid(y)
        X = np.empty((len(x), 2))
        X[:, 0] = x
        X[:, 1] = 1. - X[:, 0]
        self.T = self.exp_family.T(X)

        self.num_starts = 1
        self.K = 10


class BetaTestTinyExample(BetaTest):
    def setUpData(self):
        # get sufficient statistics
        x = (
            .1, .11,
            .7,
            .99,
        )
        X = np.empty((len(x), 2))
        X[:, 0] = x
        X[:, 1] = 1. - X[:, 0]
        self.T = self.exp_family.T(X)

        self.num_starts = 3


class BetaTestSmallExample(BetaTest):
    def setUpData(self):
        # get sufficient statistics
        x = (
            .1, .11, .1, .11, .1, .11, .1, .11, .1, .11,
            .7, .8, .9, .95, .7, .8, .9, .95,
            .7, .8, .9, .95, .7, .8, .9, .95,
            .99, .98, .97, .999
        )
        X = np.empty((len(x), 2))
        X[:, 0] = x
        X[:, 1] = 1. - X[:, 0]
        self.T = self.exp_family.T(X)

        self.num_starts = 10

    def test_eta_ML_estimates(self):
        "Test eta ML estimates."
        logging.info('Testing eta ML estimates.')
        self.options.point_estimates = True
        mixture = beta.ExpFamilyMixture(
            self.T,
            self.weights,
            self.K,
            self.exp_family,
            -np.ones(2),
            1.,
            options=self.options
        )
        # test with particular q_zs that have been causing problems
        for q_zk in (
            np.array([
                0.29749273, 0.58481013, 0.18151185, 0.31294129, 0.09686822,
                0.44975687, 0.07619981, 0.68936031, 0.47197026, 0.01709145,
                0.02711775, 0.32861889, 0.38951457, 0.19094589, 0.16494857,
                0.17816501, 0.6811572, 0.67518194, 0.42680854, 0.07046853,
                0.05905887, 0.39504161, 0.23160204, 0.26221719, 0.0650952,
                0.05591776, 0.28755008, 0.34947624, 0.10033362, 0.6189531
            ]),
            # commented out q_zs caused warnings about "Gradient and/or function calls not changing."
            #            np.array([
            #                8.63871130e-04,   1.20063955e-03,   8.63871130e-04,
            #                1.20063955e-03,   8.63871130e-04,   1.20063955e-03,
            #                8.63871130e-04,   1.20063955e-03,   8.63871130e-04,
            #                1.20063955e-03,   8.02844865e-01,   8.50321369e-01,
            #                8.69369013e-01,   8.63688717e-01,   8.02844865e-01,
            #                8.50321369e-01,   8.69369013e-01,   8.63688717e-01,
            #                8.02844865e-01,   8.50321369e-01,   8.69369013e-01,
            #                8.63688717e-01,   8.02844865e-01,   8.50321369e-01,
            #                8.69369013e-01,   8.63688717e-01,   8.11946373e-01,
            #                8.39190308e-01,   8.51861325e-01,   6.80120798e-01
            #            ]),
            #            np.array([
            #                0.00213664,  0.00285247,  0.00213664,  0.00285247,  0.00213664,
            #                0.00285247,  0.00213664,  0.00285247,  0.00213664,  0.00285247,
            #                0.54968579,  0.65841232,  0.7269619 ,  0.73684215,  0.54968579,
            #                0.65841232,  0.7269619 ,  0.73684215,  0.54968579,  0.65841232,
            #                0.7269619 ,  0.73684215,  0.54968579,  0.65841232,  0.7269619 ,
            #                0.73684215,  0.69139635,  0.71735111,  0.72877787,  0.57628613
            #            ]),
            #            np.array([
            #                7.41645164e-01,   7.39269925e-01,   7.41645164e-01,
            #                7.39269925e-01,   7.41645164e-01,   7.39269925e-01,
            #                7.41645164e-01,   7.39269925e-01,   7.41645164e-01,
            #                7.39269925e-01,   7.98735944e-02,   2.53066140e-02,
            #                4.09112455e-03,   7.48916458e-04,   7.98735944e-02,
            #                2.53066140e-02,   4.09112455e-03,   7.48916458e-04,
            #                7.98735944e-02,   2.53066140e-02,   4.09112455e-03,
            #                7.48916458e-04,   7.98735944e-02,   2.53066140e-02,
            #                4.09112455e-03,   7.48916458e-04,   1.73781525e-05,
            #                8.65900146e-05,   2.23302347e-04,   8.49141084e-08
            #            ]),
            #            np.array([
            #                0.52997879,  0.52379167,  0.52997879,  0.52379167,  0.52997879,
            #                0.52379167,  0.52997879,  0.52379167,  0.52997879,  0.52379167,
            #                0.15753093,  0.10400113,  0.05263557,  0.02712969,  0.15753093,
            #                0.10400113,  0.05263557,  0.02712969,  0.15753093,  0.10400113,
            #                0.05263557,  0.02712969,  0.15753093,  0.10400113,  0.05263557,
            #                0.02712969,  0.00592375,  0.01140258,  0.01672451,  0.00066301
            #            ]),
            np.array([
                0.00234448, 0.00311854, 0.00234448, 0.00311854, 0.00234448,
                0.00311854, 0.00234448, 0.00311854, 0.00234448, 0.00311854,
                0.52544951, 0.63461792, 0.710777, 0.7276797, 0.52544951,
                0.63461792, 0.710777, 0.7276797, 0.52544951, 0.63461792,
                0.710777, 0.7276797, 0.52544951, 0.63461792, 0.710777,
                0.7276797, 0.69394447, 0.71529741, 0.72379691, 0.593831
            ]),
            np.array([
                0.03846256, 0.04374457, 0.03846256, 0.04374457, 0.03846256,
                0.04374457, 0.03846256, 0.04374457, 0.03846256, 0.04374457,
                0.42437703, 0.47713796, 0.51621063, 0.52125131, 0.42437703,
                0.47713796, 0.51621063, 0.52125131, 0.42437703, 0.47713796,
                0.51621063, 0.52125131, 0.42437703, 0.47713796, 0.51621063,
                0.52125131, 0.48190284, 0.50374186, 0.51369768, 0.39423033
            ]),
            np.array([
                0.19474293, 0.1965585, 0.19474293, 0.1965585, 0.19474293,
                0.1965585, 0.19474293, 0.1965585, 0.19474293, 0.1965585,
                0.26903444, 0.21701202, 0.18640705, 0.18765729, 0.26903444,
                0.21701202, 0.18640705, 0.18765729, 0.26903444, 0.21701202,
                0.18640705, 0.18765729, 0.26903444, 0.21701202, 0.18640705,
                0.18765729, 0.23603519, 0.21008995, 0.19812374, 0.35505404
            ]),
            np.array([
                0.42160828, 0.41092756, 0.42160828, 0.41092756, 0.42160828,
                0.41092756, 0.42160828, 0.41092756, 0.42160828, 0.41092756,
                0.30000205, 0.3255985, 0.38768849, 0.46364489, 0.30000205,
                0.3255985, 0.38768849, 0.46364489, 0.30000205, 0.3255985,
                0.38768849, 0.46364489, 0.30000205, 0.3255985, 0.38768849,
                0.46364489, 0.65292581, 0.57249729, 0.52399792, 0.85535536
            ]),
            np.array([
                8.04020646e-01, 8.01748238e-01, 8.04020646e-01,
                8.01748238e-01, 8.04020646e-01, 8.01748238e-01,
                8.04020646e-01, 8.01748238e-01, 8.04020646e-01,
                8.01748238e-01, 3.60751411e-02, 6.88786689e-03,
                4.87303076e-04, 3.96372570e-05, 3.60751411e-02,
                6.88786689e-03, 4.87303076e-04, 3.96372570e-05,
                3.60751411e-02, 6.88786689e-03, 4.87303076e-04,
                3.96372570e-05, 3.60751411e-02, 6.88786689e-03,
                4.87303076e-04, 3.96372570e-05, 1.41173355e-07,
                1.57588400e-06, 6.51598651e-06, 4.73397905e-11
            ]),
            np.array([
                0.05977059, 0.06163495, 0.05977059, 0.06163495, 0.05977059,
                0.06163495, 0.05977059, 0.06163495, 0.05977059, 0.06163495,
                0.16752166, 0.13072282, 0.11374981, 0.1182923, 0.16752166,
                0.13072282, 0.11374981, 0.1182923, 0.16752166, 0.13072282,
                0.11374981, 0.1182923, 0.16752166, 0.13072282, 0.11374981,
                0.1182923, 0.16288209, 0.13927484, 0.12837416, 0.27975146
            ]),
        ):
            eta_0_ML, _value = mixture.q_eta.ML_estimate(mixture, q_zk)
            if False:  # make this True to create plot.
                import pylab as pl
                pl.figure()
                _theta_0_ML = mixture.exp_family.theta(eta_0_ML)
                logging.info('ML estimate for theta: %s', str(_theta_0_ML))
                mixture.plot_ll(q_zk, 2. * _theta_0_ML.max())
                pl.plot(_theta_0_ML[1], _theta_0_ML[0], 'k+', markersize=20.)
                pl.show()
                break
            assert np.isfinite(eta_0_ML).all()


class ApproxTest(unittest.TestCase):

    def sample(self, fn, approx_fn, mean, variance, num_samples):
        samples = mean - variance + \
            np.random.poisson(lam=variance, size=num_samples)
        sampled_value = fn(samples).mean()
        return sampled_value, approx_fn(mean, variance)

    def test_log_approximation_by_sampling(self):
        "Test log approximation by sampling from Poisson to estimate expectation."
        logging.info(
            'Testing log approximation by sampling from Poisson to estimate expectation.')
        num_samples = 1000
        mean = 15.
        variance = 10.
        np.random.seed(1)
        sampled, approximate = self.sample(
            np.log, beta.approx_exp_log, mean, variance, num_samples)
        #print sampled, approximate
        assert np.abs(sampled - approximate) < .01

    def test_log_gamma_approximation_by_sampling(self):
        "Test log approximation by sampling from Poisson to estimate expectation."
        logging.info(
            'Testing log approximation by sampling from Poisson to estimate expectation.')
        num_samples = 1000
        mean = 15.
        variance = 10.
        np.random.seed(1)
        sampled, approximate = self.sample(
            gammaln, beta.approx_exp_log_gamma, mean, variance, num_samples)
        #print sampled, approximate
        assert np.abs(sampled - approximate) < .3

    def calculate_exact_expectation(self, fn, approx_fn, N, p, alpha):
        """
        Test an approximating function by using alpha + a sum of N Bernoulli variables, each one parametrised by p.

        @return: Exact value, calculated approximation
        """
        expectation = 0.
        total_prob = 0.
        for k in range(N + 1):
            sum = binomial_coefficient(N, k)
            prob = sum * p**k * (1 - p)**(N - k)
            total_prob += prob
            expectation += prob * fn(alpha + k)
        assert np.allclose(1., total_prob)

        mean = N * p + alpha
        variance = N * p * (1 - p)

        return expectation, approx_fn(mean, variance)

    def test_log_approximation_exactly(self):
        "Test log approximation against exact sum of Bernoulli."
        logging.info(
            'Testing log approximation against exact sum of Bernoulli.')
        N = 5
        p = .5
        pre_compute_log_factorial(N)
        sampled, approximate = self.calculate_exact_expectation(
            np.log, beta.approx_exp_log, N, p, alpha=1.)
        #print sampled, approximate
        assert np.abs(sampled - approximate) < .01

    def test_log_gamma_approximation_exactly(self):
        "Test log gamma approximation against exact sum of Bernoulli."
        logging.info(
            'Testing log gamma approximation against exact sum of Bernoulli')
        N = 5
        p = .5
        pre_compute_log_factorial(N)
        sampled, approximate = self.calculate_exact_expectation(
            gammaln, beta.approx_exp_log_gamma, N, p, alpha=1.)
        #print sampled, approximate
        assert np.abs(sampled - approximate) < .02


if '__main__' == __name__:
    logging.basicConfig(level=logging.INFO)
    for data_cls in (
        #        BetaTestSinglePoint,
        #        BetaTestTinyExample,
        BetaTestSmallExample,
        BetaTestBigExample,
    ):
        # pass
        #        data_cls('test_integrate_out_dirichlet').debug()
        #        data_cls('test_integrate_out_stick').debug()
        #        data_cls('test_point_estimates').debug()
        data_cls('test_stick_breaking').debug()
#        data_cls('test_full_dist').debug()
    BetaTestSmallExample('test_eta_ML_estimates').debug()
    ApproxTest('test_log_approximation_by_sampling').debug()
    ApproxTest('test_log_gamma_approximation_by_sampling').debug()
    ApproxTest('test_log_approximation_exactly').debug()
    ApproxTest('test_log_gamma_approximation_exactly').debug()
    GreaterThanTest('test_greater_than').debug()
    GreaterThanTest('test_greater_than_2d').debug()
    GreaterThanTest('test_less_than_2d').debug()

    # unittest.main()
