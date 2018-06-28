#
# Copyright John Reid 2010
#

"""
Code for mixtures of beta distributions.
"""



import logging
from optparse import OptionParser
import numpy as np
import pylab as pl
import scipy.optimize as op
from scipy.special import polygamma, digamma, gamma, gammaln, betaln
from scipy.integrate import dblquad
from scipy.optimize import newton, fmin_ncg
from infpy.dp.hdpm.math import BetaDist, _permutation_by_sort, _permute
from infpy.exp import DirichletExpFamily
from infpy.convergence_test import LlConvergenceTest, check_LL_increased


digamma_1 = digamma(1.)


def digamma_inv(y):
    "@return: The inverse of the digamma function, that is x such that digamma(x) = y"
    if y >= -2.22:
        x_start = np.exp(y) + .5
    else:
        x_start = -1. / (y - digamma_1)

    def f(x):
        return digamma(x) - y

    def fprime(x):
        return polygamma(1, x)

    x_zero = newton(f, x_start, fprime, maxiter=10)
    return x_zero


def approx_exp_log(x, v):
    "@return: An approximation to the expectation of log(x), given x's mean and variance."
    return np.log(x) - v / (2 * x**2)


def approx_exp_log_gamma(x, v):
    "@return: An approximation to the expectation of log Gamma(x), given x's mean and variance."
    return gammaln(x) + .5 * v * polygamma(1, x)


def approx_exp_log_beta(x1, v1, x2, v2):
    "@return: An approximation to the expectation of log Beta(x1, x2), given x1 and x2's mean and variance."
    return approx_exp_log_gamma(x1, v1) + approx_exp_log_gamma(x2, v2) - approx_exp_log_gamma(x1 + x2, v1 + v2)


def estimate_beta_parameters_newton(tau, nu, max_iter=5000, tol=1e-12):
    "Estimate canonical beta parameters from observed sufficient statistics."
    log_p_bar = tau / nu

    def f(alpha):
        return -nu * (gammaln(alpha.sum()) - gammaln(alpha).sum() + np.dot(alpha - 1., log_p_bar))

    def f_prime(alpha):
        return -nu * (digamma(alpha.sum()) - digamma(alpha) + log_p_bar)

    def f_hess(alpha):
        return -nu * (polygamma(1, alpha.sum()) - np.diag(polygamma(1, alpha)))

    # check gradient
    if __debug__:
        from infpy.exp.test.exp_family_test import check_gradient
        x0 = np.array([2.2, 1.9])
        tol = 1e-2
        check_gradient(f, f_prime, x0, tol=tol)
        check_gradient(lambda eta: f_prime(
            eta)[0], lambda eta: f_hess(eta)[0], x0, tol=tol)
        check_gradient(lambda eta: f_prime(
            eta)[1], lambda eta: f_hess(eta)[1], x0, tol=tol)

    alpha0 = johnson_kotz_alpha_starting_point(log_p_bar)

    xopt, fopt, func_calls, grad_calls, hess_calls, warnflag = fmin_ncg(
        f, alpha0, f_prime, fhess=f_hess, full_output=True, disp=0, maxiter=max_iter)
    logging.debug(
        '\nxopt = %s\nfopt = %f\nfunc_calls = %d\ngrad_calls = %d\nhess_calls = %d\nwarnflag = %d',
        xopt, fopt, func_calls, grad_calls, hess_calls, warnflag
    )
    if 1 & warnflag:
        logging.warning('Maximum number of iterations exceeded.')
        raise RuntimeError(
            'Maximum number of iterations exceeded.\ntau = %s; nu = %f\nfopt = %f' % (tau, nu, fopt))
    elif 2 & warnflag:
        logging.warning('Gradient and/or function calls not changing.')
        #raise RuntimeError('Gradient and/or function calls not changing.')

    if not np.isfinite(xopt).all():
        raise RuntimeError(
            'xopt not finite: %s\ntau = %s; nu = %f' % (xopt, tau, nu))

    return xopt - 1.


def johnson_kotz_alpha_starting_point(log_p_bar):
    p_bar = np.exp(log_p_bar)
    _tmp = (1. - p_bar.sum())
    alpha = (_tmp + p_bar) / (2. * _tmp)
    return alpha


def estimate_beta_parameters(tau, nu, max_iter=5000, tol=1e-12):
    "Estimate the canonical beta parameters from the observed sufficient statistics."
    logging.info('tau = %s; nu = %f', tau, nu)
    log_p_bar = tau / nu
    alpha = johnson_kotz_alpha_starting_point(log_p_bar)
    for i in range(max_iter):
        last_alpha = alpha.copy()
        for k, _p in enumerate(log_p_bar):
            alpha[k] = digamma_inv(digamma(alpha.sum()) + _p)
            if not np.isfinite(alpha[k]):
                raise RuntimeError('Not finite after %d iterations' % i)
        change = np.sqrt(((last_alpha - alpha)**2).sum())
        if change < tol:
            break
    else:
        raise RuntimeError('Maximum iterations (%d) exceeded; change=%f\ntau %s; nu %f' % (
            max_iter, change, tau, nu))
    return alpha - 1.


def safe_x_log_x(x):
    "@return: x log(x) where 0 * log(0) = 0."
    x = np.asarray(x)
    l = np.zeros_like(x)
    l[x > 0] = np.log(x[x > 0])
    return x * l


def plot_density_with_R(x, weights, filename, mixture_x=None, mixture=None, adjust=.05):
    "Use R to plot density."
    import rpy2.robjects as robjects
    from rpy2.robjects import r as R
    from rpy2.robjects.packages import importr
    r_x = robjects.FloatVector(x)
    r_weights = robjects.FloatVector(weights / weights.sum())
    density_args = {
        'from': 0,
        'to': 1,
        'weights': r_weights,
        'adjust': adjust,
    }
    r_density = R['density'](r_x, **density_args)
    grdevices = importr('grDevices')
    grdevices.pdf(file='%s.pdf' % filename, width=4.652, height=3.2)
    R.par(mar=robjects.FloatVector(np.array([4, 4, 0, 0.]) + .2))
    R['plot'](r_density, ty="l", main="")
    R['rug'](len(r_x) > 1000 and R['sample'](r_x, 1000) or r_x)
    if None != mixture_x:
        if None == mixture:
            raise ValueError('Must specify mixture_x and mixture or neither')
        R['lines'](robjects.FloatVector(mixture_x),
                   robjects.FloatVector(mixture), col="red")
    grdevices.dev_off()


def plot_digamma_differences():
    min = .1
    max = 2.
    num_points = 1000
    step = (max - min) / num_points
    a, b = np.ogrid[min:max:step, step:max:step]
    pl.imshow(digamma(a) - digamma(a + b), origin='lower',
              extent=[step, max, min, max])
    pl.colorbar()
    pl.title('$\psi(a) - \psi(a+b)$')
    pl.xlabel('$b$')
    pl.ylabel('$a$')


def transform_fn_and_derivatives(f, h, fprime, hprime, fhess=None, hhess=None):
    "@return: A transformed version of f and its derivatives where h is the transform."
    def transformed(x):
        return f(h(x))

    def transformed_prime(x):
        return np.dot(fprime(h(x)), hprime(x))

    if None == fhess:
        return transformed, transformed_prime

    else:
        assert None != hprime

        def transformed_hess(x):
            deriv = hprime(x)
            return np.outer(deriv, deriv) * fhess(h(x)) + hhess(x) * fprime(h(x))
            # return hhess(x) * fprime(h(x)) + hprime(x)**2 * fhess(h(x))

        return transformed, transformed_prime, transformed_hess


def greater_than(x):
    "@return: An array containing the sum of the elements in x at higher indices."
    assert 1 <= x.ndim
    result = np.empty_like(x)
    result[-1] = 0.
    for k, c in enumerate(x[-1:0:-1]):
        result[-2 - k] = result[-1 - k] + c
    return result


def greater_than_2d(x):
    "@return: An array y_ij = sum_{k=j+1..N} x_ik."
    assert 2 == x.ndim
    y = np.empty_like(x)
    y[:, -1] = 0
    for i in range(2, x.shape[1] + 1):
        y[:, -i] = x[:, -i + 1] + y[:, -i + 1]
    return y


def less_than_2d(x):
    "@return: An array y_ij = sum_{k=1..j-1} x_ik."
    assert 2 == x.ndim
    y = np.empty_like(x)
    y[:, 0] = 0
    for i in range(1, x.shape[1]):
        y[:, i] = x[:, i - 1] + y[:, i - 1]
    return y


def check_dblquad(f):
    #epsabs = _epsrel = 1.5e-4
    result, _abserr = dblquad(
        f,
        -1.,
        np.infty,
        lambda x: -1.,
        lambda x: np.infty,
        # epsabs=epsabs,
        # epsrel=_epsrel,
    )
    # if _abserr > epsabs * 10.:
    #    raise RuntimeError('Double quadrature failed. Absolute error: %f > %f' % (_abserr, epsabs))
    assert np.isfinite(result)
    return result


def beta_conj_prior_predictive_T(T, tau, nu, A=None):
    p = np.empty(len(T))
    if None == A:
        A = beta_conj_prior_log_partition(tau, nu)
    for i, t in enumerate(T):
        A_data = beta_conj_prior_log_partition(tau + t, nu + 1.)
        p[i] = np.exp(A_data - A)
    return p


def beta_conj_prior_log_partition(tau, nu):
    "@return: The log partition function of the beta conjugate prior evaluated at the given tau and nu."
    def f(eta0, eta1):
        result = np.exp(eta0 * tau[0] + eta1 *
                        tau[1] - nu * betaln(eta0 + 1., eta1 + 1.))
        # if np.isinf(result):
        #    result = np.finfo(np.float32).max
        assert np.isfinite(result)
        return result
    return np.log(check_dblquad(f))


def beta_conj_prior_expected_A(tau, nu, A):
    "@return: The expected value of the beta log partition function under the beta conjugate prior for the given tau and nu."
    def f(eta0, eta1):
        beta_A = betaln(eta0 + 1., eta1 + 1.)
        return beta_A * np.exp(eta0 * tau[0] + eta1 * tau[1] - nu * beta_A - A)
    return check_dblquad(f)


def beta_conj_expected_eta(tau, nu, A):
    "@return: The expected eta under the beta conjugate prior for the given tau and nu."
    def f0(eta0, eta1):
        return eta0 * np.exp(eta0 * tau[0] + eta1 * tau[1] - nu * betaln(eta0 + 1., eta1 + 1.) - A)

    def f1(eta0, eta1):
        return eta1 * np.exp(eta0 * tau[0] + eta1 * tau[1] - nu * betaln(eta0 + 1., eta1 + 1.) - A)
    return np.array([check_dblquad(f0), check_dblquad(f1)])


class EtaDist(object):
    def expected_LL(self, mixture):
        exp_eta, exp_A = self.expected_values()
        assert (self.K,) == exp_A.shape
        assert (self.K, 2) == exp_eta.shape

        data_term = np.dot(mixture.x, exp_eta.T)  # NxK
        assert (mixture.N, self.K) == data_term.shape

        exp_part = data_term - exp_A  # NxK
        assert (mixture.N, self.K) == exp_part.shape

        weighted_z = (mixture.weights * mixture.q_z.params.T).T  # NxK
        assert (mixture.N, self.K) == weighted_z.shape

        weighted_term = weighted_z * exp_part  # NxK
        assert (mixture.N, self.K) == weighted_term.shape

        eta_prior = np.dot(exp_eta, mixture.tau) - mixture.nu * exp_A  # K
        assert (self.K,) == eta_prior.shape

        LL = (
            weighted_term.sum()
            + eta_prior.sum()
        )
        if hasattr(mixture, 'A_tau_nu'):
            LL -= mixture.K * mixture.A_tau_nu

        return LL


def make_ll_fns(tau, nu):
    """
    @return: ll, ll_prime, ll_hess: The log likelihood as a function of eta and its derivative and hessian.
    """
    def ll(x):
        eta = np.expm1(x)
        eta_plus_one = np.exp(x)
        rolled = np.rollaxis(eta_plus_one, -1)
        return np.dot(eta, tau) - nu * betaln(rolled[0], rolled[1])

    def ll_prime(x):
        eta_plus_one = np.exp(x)
        rolled = np.rollaxis(eta_plus_one, -1)
        return eta_plus_one * (tau - nu * (digamma(rolled) - digamma(rolled.sum())))

    def ll_hess(x):
        eta_plus_one = np.exp(x)
        rolled = np.rollaxis(eta_plus_one, -1)
        return (
            np.diag(eta_plus_one * (tau - nu *
                                    (digamma(rolled) - digamma(rolled.sum()))))
            - np.outer(eta_plus_one, eta_plus_one) * nu *
            (np.diag(polygamma(1, rolled)) - polygamma(1, rolled.sum(axis=0)))
        )

    return ll, ll_prime, ll_hess, np.expm1, np.log1p


def transform_ll_fns(ll, ll_prime, ll_hess):
    """
    Apply a transform to give our LL function a range from -infinity to +infinity.
    """
    def h(x):
        return np.expm1(x)

    def hinv(eta):
        return np.log1p(eta)

    def hprime(x):
        return np.diag(np.exp(x))

    def hhess(x):
        D = len(x)
        result = np.zeros((D, D, D))
        for d, _x in enumerate(x):
            result[d, d, d] = np.exp(_x)
        return result

    return transform_fn_and_derivatives(ll, h, ll_prime, hprime, ll_hess, hhess), h, hinv, hprime, hhess


def find_ML_eta(tau, nu, starting_eta=None):
    """
    Find ML estimate of eta for given tau and nu using generic optimisation.
    """
    ll, ll_prime, ll_hess, transform, transform_inv = make_ll_fns(tau, nu)
#    (transformed_ll, transformed_ll_prime, _transformed_ll_hess), transform, transform_inv, transform_prime, transform_hess = \
#        transform_ll_fns(ll, ll_prime, ll_hess)

    optimizer = op.fmin_cg
    #optimizer = op.fmin_bfgs
    #optimizer = op.fmin_ncg

    def to_optimize(x): return -ll(x)

    def to_optimize_prime(x): return -ll_prime(x)
    starts = (
        np.log(.5 * np.ones(2)),
        np.log(np.ones(2)),
        np.log(.1 * np.ones(2)),
    )

    if None != starting_eta:
        starts = (transform_inv(starting_eta),) + starts

    # check gradient if running debug version
    if __debug__:
        from infpy.exp.test.exp_family_test import check_gradient
        x0 = np.array([1.2, .9])
        tol = 1e-2
        check_gradient(ll, ll_prime, x0, tol=tol)
#        check_gradient(lambda x: transform(x)[0], lambda x: transform_prime(x)[0], x0)
#        check_gradient(lambda x: transform(x)[1], lambda x: transform_prime(x)[1], x0)
#        check_gradient(transformed_ll, transformed_ll_prime, x0, tol=tol)
        # check first entry in hessians
        check_gradient(lambda eta: ll_prime(
            eta)[0], lambda eta: ll_hess(eta)[0], x0, tol=tol)
        check_gradient(lambda eta: ll_prime(
            eta)[1], lambda eta: ll_hess(eta)[1], x0, tol=tol)
#        check_gradient(lambda x: transform_prime(x)[0,1], lambda x: transform_hess(x)[0,1], x0)
#        check_gradient(lambda x: transform_prime(x)[0,0], lambda x: transform_hess(x)[0,0], x0)
        #check_gradient(lambda eta: transformed_ll_prime(eta)[0], lambda eta: transformed_ll_hess(eta)[0], x0, tol=tol)

    # optimise from each starting point until we find one that gives a finite result
    for start in starts:
        xopt, fopt, func_calls, grad_calls, warnflag = optimizer(
            to_optimize,
            start,
            fprime=to_optimize_prime,
            full_output=True,
            disp=0
        )
        if 1 & warnflag:
            logging.warning('Maximum number of iterations exceeded.')
            raise RuntimeError('Maximum number of iterations exceeded.')
        elif 2 & warnflag:
            logging.warning('Gradient and/or function calls not changing.')
            #raise RuntimeError('Gradient and/or function calls not changing.')

        result = transform(xopt)

        # did we find a finite maxima?
        if np.isfinite(result).all():
            break

        # otherwise try another start
        logging.warning('Trying another start... result was %s', str(result))
    else:
        logging.warning('Maxima not finite.')

    # report maximum
    logging.debug('Maximum of %f found at %s', -fopt, result)
    logging.debug('%s made %d function calls and %d gradient calls',
                  optimizer.__name__, func_calls, grad_calls)

    # check we improved on starting point if we were given one
    if None != starting_eta:
        assert ll(starting_eta) <= ll(result) + 1e-3
        #logging.info('Improved on starting eta by %f', f(result) - f(starting_eta))

    return result, -fopt


class EtaPointEstimates(EtaDist):
    def __init__(self, K, exp_family):
        self.K = K
        self.exp_family = exp_family
        #self.eta = np.random.rand(self.K, self.exp_family.dimension) * 10. - 1.
        self.eta = np.random.lognormal(mean=0., sigma=3., size=(
            self.K, self.exp_family.dimension)) - 1.

    def _check_shapes(self):
        assert (self.K, self.exp_family.dimension) == self.eta.shape

    def update(self, mixture):
        # get point estimates for etas
        if __debug__:
            last_bound = mixture.variational_bound_piecewise()
        for k in range(self.K):
            # new estimate for eta
            #            self.eta[k] = self.ML_estimate2(mixture, mixture.q_z.params[:,k])
            # old estimate for eta
            self.eta[k], _bound = self.ML_estimate(
                mixture, mixture.q_z.params[:, k], starting_eta=self.eta[k])
#            diff = np.sqrt(((self.eta[k] - eta)**2).sum())
#            if diff > 1e-5:
#                raise RuntimeError('Different ML etas: %f; %s and %s' % (diff, eta, self.eta[k]))
            if __debug__:
                check_LL_increased(last_bound, mixture.variational_bound_piecewise(
                ), 'Component %d' % k, tolerance=1e-5)
                #logging.info('Component %d: new eta: %s', k, self.eta[k])
        assert np.isfinite(self.eta).all()

    def expected_values(self):
        "@return: Expected eta and expected A(eta)."
        return self.eta, self.exp_family.A(self.eta)

    def predictive(self, num_points=None):
        "@return: The probability of a number of points in X-space."
        if None == num_points:
            num_points = 500
        X = np.empty((num_points, 2))
        X[:, 0] = np.linspace(0., 1., num_points)
        X[:, 1] = 1. - X[:, 0]
        T = self.exp_family.T(X)
        assert T.shape == (num_points, 2)
        p = self.evaluate(T)
        return X[:, 0], p

    def evaluate(self, T):
        "@return: The probability of a number of points in X-space."
        assert 2 == len(T.shape)
        p = np.empty((self.K, len(T)))
        for k in range(self.K):
            p[k] = np.exp(np.dot(T, self.eta[k]) -
                          betaln(self.eta[k, 0] + 1., self.eta[k, 1] + 1.))
        assert (0. <= p).all()
        return p

    def entropy(self, mixture):
        return 0.

    def new_ML_estimate(self, mixture, q_zk, starting_eta=None, use_transformed_ll=True):
        """
        Uses newer LL functions. May be more accurate than older LL calculation and hence bound does not always increase.
        """
        assert (mixture.N,) == q_zk.shape
        tau, nu = mixture.tau_and_nu_for_q_zk(q_zk)
        return find_ML_eta(tau, nu)

    def permute(self, permutation):
        self.eta = _permute(self.eta, permutation, axis=0)

    def ML_estimate2(self, mixture, q_zk, max_iter=5000, tol=1e-8):
        tau, nu = mixture.tau_and_nu_for_q_zk(q_zk)
        return estimate_beta_parameters_newton(tau, nu, max_iter, tol)

    def ML_estimate(self, mixture, q_zk, starting_eta=None, use_transformed_ll=True):
        assert (mixture.N,) == q_zk.shape
        ll_fn, ll_prime, ll_hess = mixture.make_ll_fn(q_zk)

        # our transform
        def h(x):
            return np.exp(x) - 1.

        def hinv(eta):
            return np.log(eta + 1.)

        def hprime(x):
            return np.diag(np.exp(x))

        def hhess(x):
            D = len(x)
            result = np.zeros((D, D, D))
            for d, _x in enumerate(x):
                result[d, d, d] = np.exp(_x)
            return result

        g, gprime, ghess = transform_fn_and_derivatives(
            ll_fn, h, ll_prime, hprime, ll_hess, hhess)

        # check gradient
        if __debug__:
            from infpy.exp.test.exp_family_test import check_gradient
            x0 = np.array([1.2, .9])
            tol = 1e-2
            check_gradient(ll_fn, ll_prime, x0, tol=tol)
            check_gradient(lambda x: h(x)[0], lambda x: hprime(x)[0], x0)
            check_gradient(lambda x: h(x)[1], lambda x: hprime(x)[1], x0)
            check_gradient(g, gprime, x0, tol=tol)
            # check first entry in hessians
            check_gradient(lambda eta: ll_prime(
                eta)[0], lambda eta: ll_hess(eta)[0], x0, tol=tol)
            check_gradient(lambda x: hprime(
                x)[0, 1], lambda x: hhess(x)[0, 1], x0)
            check_gradient(lambda x: hprime(
                x)[0, 0], lambda x: hhess(x)[0, 0], x0)
            #check_gradient(lambda x: gprime(x)[0], lambda x: ghess(x)[0], x0)

        epsilon = 1.5e-3
        #gtol = 10.
        maxiter = None
        optimizer = op.fmin_cg
        #optimizer = op.fmin_bfgs
        #optimizer = op.fmin_ncg
        if use_transformed_ll:
            def to_optimize(x): return -g(x)

            def to_optimize_prime(x): return -gprime(x)
            #x0 = np.array(np.log([.5, .5]))
            starts = (
                np.log(.5 * np.ones(2)),
                np.log(np.ones(2)),
                np.log(.1 * np.ones(2)),
            )
            if None != starting_eta:
                starts = (hinv(starting_eta),) + starts
        else:
            def to_optimize(eta): return -ll_fn(eta)

            def to_optimize_prime(eta): return -ll_prime(eta)
            starts = (
                np.array([0., 0.]),
            )
            if None != starting_eta:
                starts = (starting_eta,) + starts

        # optimise from each starting point until we find one that gives a finite result
        for start in starts:
            xopt, fopt, func_calls, grad_calls, warnflag = optimizer(
                to_optimize,
                start,
                fprime=to_optimize_prime,
                # epsilon=epsilon,
                # gtol=gtol,
                maxiter=maxiter,
                full_output=True,
                disp=0
            )
            if 1 & warnflag:
                logging.warning('Maximum number of iterations exceeded.')
                tau, nu = mixture.tau_and_nu_for_q_zk(q_zk)
                raise RuntimeError(
                    'Maximum number of iterations exceeded.\nq_zk = %s\ntau=%s\nnu=%s' % (q_zk, tau, nu))
            elif 2 & warnflag:
                logging.warning('Gradient and/or function calls not changing.')
                #raise RuntimeError('Gradient and/or function calls not changing.')
            if use_transformed_ll:
                result = h(xopt)
            else:
                result = xopt
            # did we find a finite maxima?
            if np.isfinite(result).all():
                break
            # otherwise try another start
            logging.warning(
                'Trying another start... result was %s', str(result))
        else:
            logging.warning('Maxima not finite.')

        # report maximum
        logging.debug('Maximum of %f found at %s', -fopt, result)
        logging.debug('%s made %d function calls and %d gradient calls',
                      optimizer.__name__, func_calls, grad_calls)

        # check we improved on starting point if we were given one
        if None != starting_eta:
            starting_ll = ll_fn(starting_eta)
            result_ll = ll_fn(result)
            if starting_ll > result_ll + epsilon:
                mixture.plot_ll(q_zk, 10.)
                pl.show()
                tau, nu = mixture.tau_and_nu_for_q_zk(q_zk)
                raise RuntimeError('LL not improving: start=%f, result=%f\ntau = %s; nu = %s'
                                   % (starting_ll, result_ll, tau, nu)
                                   )
            assert starting_ll <= result_ll + epsilon
            #logging.info('Improved on starting eta by %f', ll_fn(result) - ll_fn(starting_eta))

#        tau, nu = mixture.tau_and_nu_for_q_zk(q_zk)
#        x = np.exp(tau[0])
#        raise

        return result, -fopt


class EtaFullDists(EtaDist):
    def __init__(self, K, exp_family):
        self.K = K
        self.exp_family = exp_family
        self.tau = np.ones((self.K, self.exp_family.dimension))
        self.nu = np.ones(self.K)
        self.conj_prior_A = np.zeros(self.K)
        self.exp_eta = np.zeros((self.K, self.exp_family.dimension))
        self.exp_beta_A = np.zeros(self.K)

    def permute(self, permutation):
        self.tau = _permute(self.tau, permutation, axis=0)
        self.nu = _permute(self.nu, permutation, axis=0)
        self.conj_prior_A = _permute(self.conj_prior_A, permutation, axis=0)
        self.exp_eta = _permute(self.exp_eta, permutation, axis=0)
        self.exp_beta_A = _permute(self.exp_beta_A, permutation, axis=0)

    def _check_shapes(self):
        assert (self.K, self.exp_family.dimension) == self.tau.shape
        assert (self.K,) == self.nu.shape
        assert (self.K,) == self.conj_prior_A.shape
        assert (self.K, self.exp_family.dimension) == self.exp_eta.shape
        assert (self.K,) == self.exp_beta_A.shape

    def update(self, mixture):
        self.tau[:] = mixture.tau + \
            np.dot(mixture.q_z.params.T, mixture.weighted_x)
        self.nu[:] = mixture.nu + np.dot(mixture.weights, mixture.q_z.params)
        assert np.isfinite(self.tau).all()

        # calculate some quantities we need later...
        for k in range(self.K):
            # calculate normalising constant
            self.conj_prior_A[k] = beta_conj_prior_log_partition(
                self.tau[k], self.nu[k])

            # calculate expected eta
            self.exp_eta[k] = beta_conj_expected_eta(
                self.tau[k], self.nu[k], self.conj_prior_A[k])

            # calculate expected beta A
            self.exp_beta_A[k] = beta_conj_prior_expected_A(
                self.tau[k], self.nu[k], self.conj_prior_A[k])

        assert np.isfinite(self.conj_prior_A).all()

    def expected_values(self):
        "@return: Expected eta and expected A(eta)."
        return self.exp_eta, self.exp_beta_A

    def predictive(self, num_points=None):
        "@return: The probability of a number of points in X-space."
        if None == num_points:
            num_points = 10
        X = np.empty((num_points, 2))
        margin = .1 / num_points
        X[:, 0] = np.linspace(margin, 1. - margin, num_points)
        X[:, 1] = 1. - X[:, 0]
        T = self.exp_family.T(X)
        assert T.shape == (num_points, 2)
        p = np.empty((self.K, num_points))
        for k in range(self.K):
            p[k] = beta_conj_prior_predictive_T(
                T, self.tau[k], self.nu[k], A=self.conj_prior_A[k])
        return X[:, 0], p

    def evaluate(self, T):
        "@return: The probability of a number of points in X-space."
        assert 2 == len(T.shape)
        p = np.empty((self.K, len(T)))
        for k in range(self.K):
            for t, _T in enumerate(T):
                p[k, t] = np.exp(beta_conj_prior_log_partition(
                    self.tau[k] + _T, self.nu[k] + 1.) - self.conj_prior_A[k])
        assert (0. <= p).all()
        return p

    def entropy(self, mixture):
        result = -(
            np.dot(self.exp_eta, self.tau.T).sum()
            - (self.nu * self.exp_beta_A).sum()
            - self.conj_prior_A.sum()
        )
        assert np.isfinite(result)
        return result


class DirichletPiDist(object):
    "Note this uses a Dirichlet prior parameterised by all ones."

    def __init__(self, mixture):
        self.K = mixture.K
        self.params = np.ones(self.K)

    def _check_shapes(self):
        assert (self.K,) == self.params.shape

    def update(self, mixture):
        self.params[:] = np.dot(
            mixture.weights, mixture.q_z.params) + mixture.alpha
        assert np.isfinite(self.params).all()

    def E(self):
        "@return: Expectation of pi."
        return self.params / self.params.sum()

    def log_G(self):
        "@return: log of geometric expectation of pi."
        return digamma(self.params) - digamma(self.params.sum())

    def expected_LL(self, mixture):
        return gammaln(self.K * mixture.alpha) - self.K * gammaln(mixture.alpha) + self.log_G().sum() * (mixture.alpha - 1.)

    def entropy(self, mixture):
        alpha_0 = self.params.sum()
        return (
            gammaln(self.params).sum() - gammaln(alpha_0)
            + (alpha_0 - self.K) * digamma(alpha_0)
            - np.dot(self.params - 1., digamma(self.params))
        )

    def permute(self, permutation):
        self.params = _permute(self.params, permutation, axis=0)


class StickBreakingPiDist(object):
    "A stick-breaking prior for pi."

    def __init__(self, mixture):
        self.K = mixture.K
        #self._update(np.zeros((self.K,)), mixture.alpha)
        self.update(mixture)

    def _check_shapes(self):
        assert (self.K,) == self._pi_bar.a.shape
        assert (self.K,) == self._pi_bar.b.shape

    def _update(self, counts, alpha):
        assert (self.K,) == counts.shape
        self._pi_bar = BetaDist(counts + 1., greater_than(counts) + alpha)
        self._1_minus_pi_bar = BetaDist(self._pi_bar.b, self._pi_bar.a)
        self.G_pi = np.empty((self.K,))
        term = 1.
        for k, (g, g_) in enumerate(zip(self._pi_bar.G, self._1_minus_pi_bar.G)):
            self.G_pi[k] = g * term
            term *= g_
        assert (self.K,) == self.G_pi.shape

    def update(self, mixture):
        self._update(
            np.dot(mixture.weights, mixture.q_z.params), mixture.alpha)

    def E(self):
        "@return: Expectation of pi."
        result = self._pi_bar.E.copy()
        E_1_minus = self._1_minus_pi_bar.E
        accum = 1.
        for k in range(1, self.K - 1):
            accum *= E_1_minus[k - 1]
            result[k] *= accum
        result[-1] = 1. - result[:-1].sum()
        return result

    def log_G(self):
        "@return: log of geometric expectation of pi."
        return np.log(self.G_pi)

    def expected_LL(self, mixture):
        return - self.K * betaln(1., mixture.alpha) + (mixture.alpha - 1.) * (np.log(self._1_minus_pi_bar.G)).sum()

    def entropy(self, mixture):
        return self._pi_bar.H.sum()

    def permute(self, permutation):
        a_permuted = _permute(self._pi_bar.a, permutation, axis=0)
        b_permuted = _permute(self._pi_bar.b, permutation, axis=0)
        self._pi_bar = BetaDist(a_permuted, b_permuted)
        self._1_minus_pi_bar = BetaDist(self._pi_bar.b, self._pi_bar.a)
        self.G_pi = _permute(self.G_pi, permutation, axis=0)


class StickBreakingVarDist(object):

    def __init__(self, E_counts, E_alpha):
        """
        Create a variational distribution over pi_bar and pi given the expectation of the counts
        and the expectation of the stick-breaking parameter, alpha.
        """
        self._pi_bar = BetaDist(
            E_counts + 1., greater_than(E_counts) + E_alpha)
        self._1_minus_pi_bar = BetaDist(self._pi_bar.b, self._pi_bar.a)

    def _calc_G_pi(self):
        "Calculate the geometric expectation of pi."
        G_pi_bar = self._pi_bar.G
        G_1_minus_pi_bar = self._1_minus_pi_bar.G
        G_pi = np.empty_like(G_pi_bar)
        term = 1.
        for k, (g, g_) in enumerate(zip(G_pi_bar, G_1_minus_pi_bar)):
            G_pi[k] = g * term
            term *= g_
        return G_pi

    K = property(lambda s: s._pi_bar.a.shape[0], None, None,
                 'Limit on components in variational distribution.')
    pi_bar = property(lambda s: s._pi_bar, None, None,
                      'Beta distribution over pi_bar.')
    G_pi = property(_calc_G_pi, None, None, 'Geometric expectation of pi.')


class ZDist(object):

    def __init__(self, mixture):
        self.K = mixture.K
        self.N = mixture.N
        self.params = np.random.dirichlet(np.ones(self.K), size=self.N)

    def _check_shapes(self):
        assert (self.N, self.K) == self.params.shape

    def _update_from_weights(self, weights):
        assert np.isfinite(weights).all()
        exp_weights = np.exp(weights)
        #assert np.isfinite(exp_weights).all(), 'weights = %s\nexp_weights = %s' % (weights, exp_weights)
        # replace infinities with large numbers
        exp_weights[float('infinity') == exp_weights] = 1e300
        # exp_weights[exp_weights.sum(axis=1) == 0.] = 1./self.K # if we have zeros everywhere for a datum, spread over all components
        self.params[:] = (exp_weights.T / exp_weights.sum(axis=1)).T
        assert np.isfinite(self.params).all()
        assert np.allclose(self.params.sum(axis=1.), 1.)

    def update(self, mixture):
        exp_eta, exp_A = mixture.q_eta.expected_values()
        weights = np.dot(mixture.x, exp_eta.T) - exp_A + mixture.q_pi.log_G()
        self._update_from_weights(weights)

    def expected_LL(self, mixture):
        return np.dot(np.dot(mixture.weights, self.params), mixture.q_pi.log_G())

    def entropy(self, mixture):
        weighted_params = (mixture.weights * self.params.T).T
        return - safe_x_log_x(weighted_params).sum() - safe_x_log_x(1. - mixture.weights).sum()

    def permute(self, permutation):
        self.params = _permute(self.params, permutation, axis=1)


class ZIntegrateOutDirichletDist(ZDist):
    def __init__(self, mixture):
        super(type(self), self).__init__(mixture)

    def update(self, mixture):
        weighted_params = (mixture.weights * self.params.T).T
        V_z = ((self.params * (1. - self.params)).T * (mixture.weights ** 2)).T
        z_slash_n = weighted_params.sum(axis=0) - weighted_params
        V_z_slash_n = V_z.sum(axis=0) - V_z
        exp_eta, exp_A = mixture.q_eta.expected_values()
        weights = (
            np.dot(mixture.x, exp_eta.T)
            - exp_A
            + approx_exp_log(mixture.alpha + z_slash_n, V_z_slash_n)
        )
        # 1/0
        self._update_from_weights(weights)

#        V_z = self.params * (1. - self.params)
#        z_slash_n = self.params.sum(axis=0) - self.params
#        V_z_slash_n = V_z.sum(axis=0) - V_z
#        exp_eta, exp_A = mixture.q_eta.expected_values()
#        weights = (
#            mixture.weights * (
#                np.dot(mixture.x, exp_eta.T) - exp_A
#            ).T
#        ).T + approx_exp_log(mixture.alpha + z_slash_n, V_z_slash_n)
#        #1/0
#        self._update_from_weights(weights)

    def expected_LL(self, mixture):
        alpha_dot = mixture.alpha * self.K
        weighted_params = (mixture.weights * self.params.T).T
        V_z = ((self.params * (1. - self.params)).T * (mixture.weights ** 2)).T
        return (
            gammaln(alpha_dot)
            - self.K * gammaln(mixture.alpha)
            + approx_exp_log_gamma(mixture.alpha +
                                   weighted_params.sum(axis=0), V_z.sum(axis=0)).sum()
            - gammaln(alpha_dot + mixture.weights_sum)
        )

#        alpha_dot = mixture.alpha * self.K
#        V_z = self.params * (1. - self.params)
#        return (
#            gammaln(alpha_dot)
#            - self.K * gammaln(mixture.alpha)
#            + approx_exp_log_gamma(mixture.alpha + self.params.sum(axis=0), V_z.sum(axis=0)).sum()
#            - gammaln(alpha_dot + mixture.N)
#        )

    def E_pi(self, mixture):
        weighted_params = (mixture.weights * self.params.T).T
        counts = weighted_params.sum(axis=0) + mixture.alpha
        return counts / counts.sum()


class ZIntegrateOutStickDist(ZDist):
    def __init__(self, mixture):
        super(type(self), self).__init__(mixture)

    def update(self, mixture):
        weighted_params = (mixture.weights * self.params.T).T
        V_z = ((self.params * (1. - self.params)).T * (mixture.weights ** 2)).T
        z_slash_n = weighted_params.sum(axis=0) - weighted_params
        V_z_slash_n = V_z.sum(axis=0) - V_z
        z_slash_n_greater = greater_than_2d(z_slash_n)
        V_z_slash_n_greater = greater_than_2d(V_z_slash_n)
        z_slash_n_ge = z_slash_n + z_slash_n_greater
        V_z_slash_n_ge = V_z_slash_n + V_z_slash_n_greater
        log_z_g = approx_exp_log(mixture.alpha + z_slash_n, V_z_slash_n)
        log_z_ge = approx_exp_log(
            mixture.alpha + mixture.beta + z_slash_n_ge, V_z_slash_n_ge)
        log_z_l = less_than_2d(
            approx_exp_log(mixture.beta + z_slash_n_greater,
                           V_z_slash_n_greater)
            - log_z_ge
        )
        exp_eta, exp_A = mixture.q_eta.expected_values()
        weights = (
            np.dot(mixture.x, exp_eta.T)
            - exp_A
            + log_z_g
            - log_z_ge
            + log_z_l
        )
        self._update_from_weights(weights)

    def expected_LL(self, mixture):
        weighted_params = (mixture.weights * self.params.T).T
        V_z = ((self.params * (1. - self.params)).T * (mixture.weights ** 2)).T
        return (
            approx_exp_log_beta(
                mixture.alpha + weighted_params.sum(axis=0), V_z.sum(axis=0),
                mixture.beta +
                greater_than_2d(weighted_params).sum(
                    axis=0), greater_than_2d(V_z).sum(axis=0),
            )
            - betaln(mixture.alpha, mixture.beta)
        ).sum()

    def E_pi(self, mixture):
        weighted_params = (mixture.weights * self.params.T).T
        counts = weighted_params.sum(axis=0) + mixture.alpha
        return counts / counts.sum()


class ExpFamilyMixture(object):

    def __init__(self, x, weights, K, exp_family, tau, nu, options):
        self.set_x(x, weights)
        self.K = K
        self.exp_family = exp_family
        self.tau = tau
        self.nu = nu
        if not options.no_prior:
            self.A_tau_nu = beta_conj_prior_log_partition(self.tau, self.nu)
        self.alpha = options.alpha
        self.beta = options.beta

        if options.integrate_pi:
            logging.debug('Integrating out pi')
            if options.stick_breaking:
                logging.debug("Using stick-breaking prior for pi.")
                self.q_z = ZIntegrateOutStickDist(self)
            else:
                logging.debug("Using Dirichlet prior for pi.")
                self.q_z = ZIntegrateOutDirichletDist(self)
        else:
            logging.debug('Not integrating out pi')
            self.q_z = ZDist(self)
            if options.stick_breaking:
                logging.debug("Using stick-breaking prior for pi.")
                self.q_pi = StickBreakingPiDist(self)
            else:
                logging.debug("Using Dirichlet prior for pi.")
                self.q_pi = DirichletPiDist(self)

        if options.point_estimates:
            logging.debug(
                "Using point estimates for components' eta parameters.")
            self.q_eta = EtaPointEstimates(self.K, self.exp_family)
        else:
            logging.debug(
                "Using full variational distribution for components' eta parameters.")
            self.q_eta = EtaFullDists(self.K, self.exp_family)
            self.q_eta.update(self)

        self._check_shapes()

    def set_x(self, x, weights):
        self.x = x
        self.weights = weights
        self.weighted_x = (x.T * weights).T
        self.N = len(self.x)
        self.weights_sum = self.weights.sum()

    def _check_shapes(self):
        assert self.x.ndim == 2
        assert len(self.x) == self.N
        assert (self.x.shape[1],) == self.tau.shape
        assert len(self.weights) == len(self.x)
        assert self.x.shape == self.weighted_x.shape
        self.q_eta._check_shapes()
        self.q_z._check_shapes()
        if hasattr(self, 'q_pi'):
            self.q_pi._check_shapes()

    def E_pi(self):
        if hasattr(self, 'q_pi'):
            return self.q_pi.E()
        else:
            return self.q_z.E_pi(self)

    def tau_and_nu_for_q_zk(self, q_zk):
        weighted_z = self.weights * q_zk  # (N,)
        assert (self.N,) == weighted_z.shape

        nu = weighted_z.sum() + self.nu  # ()
        assert isinstance(nu, float)

        suff_stats = (weighted_z * self.x.T).sum(axis=1)  # (D,)
        assert (self.exp_family.dimension,) == suff_stats.shape

        tau = suff_stats + self.tau  # (D,)
        assert (self.exp_family.dimension,) == tau.shape

        return tau, nu

    def make_ll_fn(self, q_zk):
        "@return: LL, derivative and hessian as function of eta."

        tau, nu = self.tau_and_nu_for_q_zk(q_zk)

        def ll_fn(eta):
            if (eta <= -1.).any():
                return -np.infty

            if np.isinf(eta).any():
                return np.infty

            eta_shape = eta.shape[:-1]
            "The shape of the eta without the exponential family dimension."

            A = self.exp_family.A(eta)
            assert eta_shape == A.shape

            result = np.dot(eta, tau) - nu * A
            assert eta_shape == result.shape

            return result

        def ll_prime(eta):
            if (eta <= -1.).any() or np.isinf(eta).any():
                return np.zeros_like(eta)

            exp_T = self.exp_family.exp_T(eta)

            return tau - nu * exp_T

        def ll_hess(eta):
            if (eta <= -1.).any() or np.isinf(eta).any():
                return np.zeros((len(eta), len(eta)))

            cov_T = self.exp_family.cov_T(eta)

            return - nu * cov_T

        return ll_fn, ll_prime, ll_hess

    def plot_ll(self, q_zk, max):
        assert np.isfinite(max)
        num_points = 100
        step_size = max / float(num_points)
        eta = np.rollaxis(np.array(
            np.mgrid[step_size - 1:max - 1:step_size, step_size - 1:max - 1:step_size]), 0, 3)
        ll_fn, _ll_prime, _ll_hess = self.make_ll_fn(q_zk)
        ll = ll_fn(eta)

        pl.clf()
        pl.imshow(ll, origin='lower', extent=[step_size, max, step_size, max])
        pl.colorbar()
        pl.xlim(step_size, max)
        pl.ylim(step_size, max)

    def variational_bound_piecewise(self):
        "@return: The variational bound on p(D) piece-wise."
        if hasattr(self, 'q_pi'):
            return np.array([
                # part from q_eta
                self.q_eta.expected_LL(self),
                # part from z_n drawn from pi
                self.q_z.expected_LL(self),
                # part from prior on pi - this is fixed as long as alpha=1.
                self.q_pi.expected_LL(self),
                # entropy of q_eta
                self.q_eta.entropy(self),
                # entropy of z
                self.q_z.entropy(self),
                # entropy of q_pi
                self.q_pi.entropy(self),
            ])
        else:
            return np.array([
                # part from q_eta
                self.q_eta.expected_LL(self),
                # part from z_n drawn from pi
                self.q_z.expected_LL(self),
                # entropy of q_eta
                self.q_eta.entropy(self),
                # entropy of z
                self.q_z.entropy(self),
            ])

    def variational_bound(self):
        "@return: The variational bound on p(D)."
        return self.variational_bound_piecewise().sum()

    def update(self):
        level = logging.DEBUG
        logging.log(level, 'Updating q_z')
        self.q_z.update(self)
        logging.log(level, 'Reordering components')
        self.reorder_components()
        if hasattr(self, 'q_pi'):
            logging.log(level, 'Updating q_pi')
            self.q_pi.update(self)
        logging.log(level, 'Updating q_eta')
        self.q_eta.update(self)
        logging.log(level, 'Checking shapes')
        self._check_shapes()

    def reorder_components(self):
        # calculate the permutation required to put the topics in order of size.
        permutation = _permutation_by_sort(
            np.dot(self.weights, self.q_z.params))

        # don't bother re-ordering anything if permutation is the identity
        if (np.arange(self.K) != permutation).any():
            self.q_z.permute(permutation)
            self.q_eta.permute(permutation)
            if hasattr(self, 'q_pi'):
                self.q_pi.permute(permutation)

    def update_to_convergence(self, eps=1e-4, min_iter=10, max_iter=50):
        test = LlConvergenceTest(
            eps=eps, should_increase=True, use_absolute_difference=True)
        last_bound = None
        for i in range(max_iter):
            if None == last_bound:
                last_bound = self.variational_bound_piecewise()
                test(last_bound.sum())
            self.update()
            if i + 1 >= min_iter and test(last_bound.sum()):
                logging.info('Log likelihood has converged : stopping.')
                break

    def plot(self, num_points=None, log=False, density=None, legend=True, scale=True):
        "Plot the mixture's distribution."
        if log:
            plot_fn = pl.semilogy
        else:
            plot_fn = pl.plot
        x, p = self.q_eta.predictive(num_points=num_points)
        E_pi = self.E_pi()
        for k, (_p, pi) in enumerate(zip(p, E_pi)):
            if scale:
                y = pi * _p
            else:
                y = _p
            plot_fn(x, y, label='%d : %.2f' % (k, pi * self.N))
        mixture = np.dot(E_pi, p)
        plot_fn(x, mixture, 'k-', label='mixture')
        if density:
            pl.plot(x, density(x), 'k:', label='density estimate')
        if legend:
            pl.legend(loc='upper center')
        return x, mixture

    def evaluate(self, x):
        "@return: The likelihood of the xs under the model."
        p = self.q_eta.evaluate(x)
        E_pi = self.E_pi()
        return np.dot(E_pi, p)


def add_options(parser):
    "Add options for the mixture to the parser."
    parser.add_option(
        "-K",
        dest="K",
        help="Number of components in the mixture.",
        type='int',
        default=3,
    )
    parser.add_option(
        "--tolerance",
        dest="tolerance",
        help="Tolerance in variational bound before stopping.",
        type='float',
        default=1e-2,
    )
    parser.add_option(
        "--plot-freq",
        dest="plot_freq",
        help="Plot the distribution every so many iterations.",
        type='int',
        default=10,
    )
    parser.add_option(
        "--max-iter",
        dest="max_iter",
        help="Maximum number of iterations.",
        type='int',
        default=60,
    )
    parser.add_option(
        "--min-iter",
        dest="min_iter",
        help="Minimum number of iterations.",
        type='int',
        default=10,
    )
    parser.add_option(
        "--point-estimates",
        dest="point_estimates",
        help="Use point estimates.",
        action="store_true",
        default=False,
    )
    parser.add_option(
        "--stick-breaking",
        dest="stick_breaking",
        help="Use stick breaking prior for pi.",
        action="store_true",
        default=False,
    )
    parser.add_option(
        "--integrate-pi",
        dest="integrate_pi",
        help="Integrate pi out of the model.",
        action="store_true",
        default=False,
    )
    parser.add_option(
        "--no-prior",
        help="Don't use a prior on the component parameters.",
        action="store_true",
        default=False,
    )
    parser.add_option(
        "--alpha",
        dest="alpha",
        help="Strength of Dirichlet or stick-breaking prior on pi.",
        type='float',
        default=1.,
    )
    parser.add_option(
        "--beta",
        dest="beta",
        help="Hyper-parameter for stick-breaking prior on pi.",
        type='float',
        default=1.,
    )


def get_default_options():
    "@return: The default options."
    parser = OptionParser()
    add_options(parser)
    return parser.get_default_values()


def integrate_mixture(mixture):
    from scipy.integrate import quad

    def f(x):
        T = np.log([[x, 1. - x]])
        p = mixture.evaluate(T)
        assert (1,) == p.shape
        return p[0]

    result, abserr = quad(
        func=f,
        a=0.,
        b=1.,
        epsabs=1.5e-03,
        epsrel=1.5e-03,
        limit=50,
    )
    return result, abserr


def does_distribution_integrate_to_1(mixture):
    result, abserr = integrate_mixture(mixture)
    return 1. - abserr <= result and result <= 1. + abserr


if '__main__' == __name__:
    logging.basicConfig(level=logging.INFO)

    parser = OptionParser()
    add_options(parser)
    options, args = parser.parse_args()

    from cookbook.pylab_utils import set_rcParams_for_latex, get_fig_size_for_latex
    set_rcParams_for_latex()
    fig_size = get_fig_size_for_latex(345)
    pl.rcParams['figure.figsize'] = fig_size

    pl.close('all')

    for seed in range(1, 3):
        logging.info('Seeding numpy with %d', seed)
        np.random.seed(seed)

        def plot_digamma_differences():
            # plot the digamma differences to get a handle on what they look like
            set_rcParams_for_latex()
            pl.figure()
            plot_digamma_differences()
            pl.savefig('digamma-differences.eps')
            pl.savefig('digamma-differences.png')
            return

        def test_exp_family():
            # the exponential family we are dealing with
            exp_family = DirichletExpFamily(k=2)

            # some test data
            x = (
                .1, .11, .1, .11, .1, .11, .1, .11, .1, .11,
                .7, .8, .9, .95, .7, .8, .9, .95,
                .7, .8, .9, .95, .7, .8, .9, .95,
                .99, .98, .97, .999
            )
            X = np.empty((len(x), 2))
            X[:, 0] = x
            X[:, 1] = 1. - X[:, 0]

            # get sufficient statistics
            T = exp_family.T(X)

            # create a mixture of exponential family distributions
            mixture = ExpFamilyMixture(
                T, options.K, exp_family, -np.ones(2), 1., options=options)

            # plot original mixture
            pl.figure()
            mixture.plot()
            pl.title('Original mixture.')

            def plot(iter):
                logging.info(
                    'Plotting distribution after iteration %d' % (iter))
                pl.figure()
                mixture.plot()
                pl.title('Mixture after iteration %d.' % (iter))

            # update mixture
            test = LlConvergenceTest(
                eps=options.tolerance, should_increase=True, use_absolute_difference=True)
            bound = mixture.variational_bound()
            test(bound)
            for _i in range(options.max_iter):
                mixture.update()
                bound = mixture.variational_bound()
                logging.info(
                    'Iteration %d: Variational bound = %f', _i + 1, bound)
                if test(bound) and _i + 1 >= options.min_iter:
                    plot(_i + 1)
                    break
                if options.plot_freq - 1 == _i % options.plot_freq:
                    plot(_i + 1)

            mixture._check_shapes()
            return mixture

        #mixture = test_exp_family()

        def examine_taus(mixture):
            # the exponential family we are dealing with
            exp_family = DirichletExpFamily(k=2)

            for tau in (
                np.log([.5, .5]),
                np.log([.7, .3]),
                np.zeros(2),
                np.ones(2),
                -2. * np.ones(2),
                -np.ones(2),
            ):
                pl.figure()
                mixture.tau = tau
                q_z = np.zeros(mixture.N)
                eta_0_ML, _value = mixture.eta_ML_estimate(q_z)
                if np.isfinite(eta_0_ML).all():
                    theta_0_ML = exp_family.theta(eta_0_ML)
                    mixture.plot_ll(q_z, 2. * theta_0_ML.max())
                    pl.plot(theta_0_ML[1], theta_0_ML[0], 'k+', markersize=20.)
                else:
                    mixture.plot_ll(q_z, 10.)
                pl.title('tau = %s' % tau)

        def test_exp_family_plot():

            # the exponential family we are dealing with
            exp_family = DirichletExpFamily(k=2)

            pl.close('all')
            pl.figure()
            exp_family.plot(np.array([.5, .5]), color='r')
            exp_family.plot(np.array([5, 1]), color='g')
            exp_family.plot(np.array([1, 3]), color='blue')
            exp_family.plot(np.array([2, 2]), color='purple')
            exp_family.plot(np.array([2, 5]), color='black')
            pl.ylim(0, 2.6)
            pl.legend()
            pl.show()

        # test_exp_family_plot()

    pl.show()
