#
# Copyright John Reid 2006
#


import scipy
import scipy.linalg
import scipy.optimize
import numpy
import math
import sys
import warnings

from sum_kernel import *
from se_kernel import *

from numpy import matrix, array, zeros, log, diagonal, trace, arange, asarray, mean, sqrt


class GaussianProcess(object):
    """
    A Gaussian process.

    Following notation in section 2.2 of `Gaussian Processes for Machine Learning`__ by Rasmussen and Williams.

    __ http://www.amazon.co.uk/Gaussian-Processes-Learning-Adaptive-Computation/dp/026218253X/
    """

    def __init__(self, X, y, k):
        """Initialise Gaussian process

        X: training data points
        y: training outputs
        k: covariance function
        """
        self.k = k
        self.reestimate(X, y)

    def reestimate(self, X, y):
        """
        Reestimate process given the data
        """
        from numpy import matrix
        if len(X) != len(y):
            raise RuntimeError("""Supplied %d data and %d target values. """
                               """These must be equal.""" % (len(X), len(y)))
        self.X = X
        self.y = matrix(y).T
        self.n = self.y.shape[0]
        self._update()

    def calc_covariance(self, x1, x2=None):
        """
        Calculate covariance matrix of x1 against x2

        if x2 is None calculate symmetric matrix
        """
        import types
        symmetric = x2 is None  # is it symmetric?
        if symmetric:
            x2 = x1

        def f(i1, i2):
          return self.k(x1[i1], x2[i2], symmetric and i1 == i2)
        return \
            infpy.matrix_from_function(
                f,
                (len(x1), len(x2)),
                numpy.float64,
                symmetric)

    def calc_covariance_derivative(self, i, x1, x2=None):
        """
        Calculate derivative of covariance matrix of x1 against x2 w.r.t param i

        if x2 is None calculate symmetric matrix
        """
        import types
        symmetric = x2 is None  # is it symmetric?
        if symmetric:
            x2 = x1
        deriv = self.k.derivative_wrt_param(i)

        def f(i1, i2): return deriv(x1[i1], x2[i2], symmetric and i1 == i2)
        return \
            infpy.matrix_from_function(
                f,
                (len(x1), len(x2)),
                numpy.float64,
                symmetric)

    def _update(self):
        """
        Calculate those terms for prediction that do not depend on predictive
        inputs.
        """
        from numpy.linalg import cholesky, solve, LinAlgError
        from numpy import transpose, eye, matrix
        import types
        self._K = self.calc_covariance(self.X)
        if not self._K.shape[0]:  # we didn't have any data
            self._L = matrix(zeros((0, 0), numpy.float64))
            self._alpha = matrix(zeros((0, 1), numpy.float64))
            self.LL = 0.
        else:
            try:
                self._L = matrix(cholesky(self._K))
            except LinAlgError, detail:
                raise RuntimeError("""Cholesky decomposition of covariance """
                                   """matrix failed. Your kernel may not be positive """
                                   """definite. Scipy complained: %s""" % detail)
            self._alpha = solve(self._L.T, solve(self._L, self.y))
            self.LL = (
                - self.n * math.log(2.0 * math.pi)
                - (self.y.T * self._alpha)[0, 0]
            ) / 2.0
        # print self.LL
        # import IPython; IPython.Debugger.Pdb().set_trace()
        self.LL -= log(diagonal(self._L)).sum()
        # print self.LL
        # print 'Done updating'

    def predict(self, x_star):
        """
        Predict the process's values on the input values

        @arg x_star: Prediction points

        @return: ( mean, variance, LL )
        where mean are the predicted means, variance are the predicted
        variances and LL is the log likelihood of the data for the given
        value of the parameters (i.e. not integrating over hyperparameters)
        """
        from numpy.linalg import solve
        import types
        # print 'Predicting'
        if 0 == len(self.X):
            f_star_mean = matrix(zeros((len(x_star), 1), numpy.float64))
            v = matrix(zeros((0, len(x_star)), numpy.float64))
        else:
            k_star = self.calc_covariance(self.X, x_star)
            f_star_mean = k_star.T * self._alpha
            if 0 == len(x_star):  # no training data
                v = matrix(zeros((0, len(x_star)), numpy.float64))
            else:
                v = solve(self._L, k_star)
        V_f_star = self.calc_covariance(x_star) - v.T * v
        # print 'Done predicting'
        # import IPython; IPython.Debugger.Pdb().set_trace()
        return (f_star_mean, V_f_star, self.LL)

    def set_params(self, params):
        """Set the parameters of the covariance function."""
        from scipy.linalg import norm
        self.k.set_params(params)
        self._update()  # params have changed so recalculate
        # if norm( params - self.k.get_parameters() ) > 1e-12:
        #       # print 'Params changed:', params, self.k.get_parameters()
        #       self.k.set_parameters( params )
        #       self._update() # they have so recalculate


def gp_1D_predict(
        gp,
        num_steps=100,
        x_min=None,
        x_max=None,
        new_figure=True,
        show_y=True,
        show_variance=True
):
    """Predict and plot the GP's predictions over the range provided"""
    from pylab import figure, plot
    # print gp.X
    if x_min is None:
        x_min = min(x[0] for x in gp.X)
    if x_max is None:
        x_max = max(x[0] for x in gp.X)
    x_max = float(x_max)
    x_min = float(x_min)
    test_x = gp_1D_X_range(x_min, x_max, (x_max - x_min) / num_steps)
    (mean, variance, LL) = gp.predict(test_x)
    if new_figure:
        figure()
    if show_variance:
        gp_plot_prediction(test_x, mean, variance)
    else:
        gp_plot_prediction(test_x, mean, None)
    if show_y:
        plot(gp.X, gp.y, 'rs')
    gp_title_and_show(gp)


class GP_LL_fn(object):
    """GP LL as function of kernel parameters"""

    def __init__(self, gp):
      self.gp = gp

    def __call__(self, params):
        if not self.gp.k.supports(params):  # parameters not supported
            return math.log(1e-300)
        self.gp.set_params(params)
        result = self.gp.LL
        param_ll = sum(
            prior.log_pdf(param)
            for prior, param in zip(self.gp.k.param_priors, params)
            if prior is not None
        )
        result += param_ll
        return result


class GP_LL_deriv_fn(object):
    """:math:`\\frac{d\\textrm{LL}}{d\\textrm{params}}` as function of kernel parameters

    Relies on the parameters already having been set. I.e. __call__ ignores
    its arguments!
    """

    def __init__(self, gp, needs_to_set_params):
        """
        gp: the gaussian process
        needs_to_set_params: do we need to set the params on every call?
        """
        self.gp = gp
        self.needs_to_set_params = needs_to_set_params

    def __call__(self, params):
        from numpy import zeros_like
        gradient = zeros_like(params)
        if self.gp.k.supports(params):  # parameters supported?
            if self.needs_to_set_params:
                self.gp.set_params(params)
            else:
                # we can rely on the parameters already being set by the callback
                # in gp_learn_hyperparameters
                assert infpy.norm2(params - asarray(self.gp.k.params)) < 1e-10
            K_inv = infpy.lu_inv(self.gp._L)
            term = self.gp._alpha * self.gp._alpha.T - K_inv
            for i in range(len(params)):
                cov_deriv = self.gp.calc_covariance_derivative(i, self.gp.X)
                t = trace(term * cov_deriv)
                gradient[i] = t / 2.0
                # if None we have uniform prior
                if self.gp.k.param_priors[i] is not None:
                    gradient[i] += self.gp.k.param_priors[i].dlog_pdf_dx(
                        params[i])
            # print self.gp.k._params, 'gradient:', gradient
        return gradient


def gp_learn_hyperparameters(gp, initial_guess=None, *args, **kwds):
    """Run optimisation algorithm on GP's kernel parameters

    Uses conjugate gradient descent
    """
    ll = GP_LL_fn(gp)
    ll_deriv = GP_LL_deriv_fn(gp, needs_to_set_params=False)

    def to_minimise(p): return -ll(p)

    def to_minimise_derivative(p): return -ll_deriv(p)
    if False:  # test gradients
        check_gradients(
            to_minimise,
            to_minimise_derivative,
            [p for p in gp.k.params]
        )
    if None == initial_guess:
        initial_guess = [p for p in gp.k.params]
    best_params, fopt, gopt, Hopt, func_calls, grad_calls, warnflag = \
        scipy.optimize.fmin_bfgs(
            to_minimise,
            initial_guess,
            to_minimise_derivative,
            callback=gp.set_params,
            full_output=True,
            *args,
            **kwds
        )
    gp.set_params(best_params)
    if 1 == warnflag:
        raise RuntimeError(
            'GP optimize hyperparameters: Maximum number of iterations exceeded.'
        )
    elif 2 == warnflag:
        warnings.warn(
            'GP optimize hyperparameters: Gradient and/or function calls not changing.'
        )
    elif warnflag:
        raise RuntimeError(
            'GP optimize hyperparameters: Unknown warning from optimize routine.'
        )


def gp_1D_X_range(xmin, xmax, step=1.):
    """
    @return: An array of points between xmin and xmax with the given step size. The shape of the array
    will be (N, 1) where N is the number of points as this is the format the GP code expects.

    Implementation uses numpy.arange so last element of array may be > xmax.
    """
    result = numpy.arange(xmin, xmax, step)
    result.resize((len(result), 1))
    return result


def gp_zero_mean(y):
    """Return the data shifted to make the mean zero"""
    return y - mean(y)


def gp_sample_at(
        gp,
        x
):
    """Samples from a gaussian process

    x is the value to sample at
    """
    import scipy.stats
    (mean, sigma, LL) = gp.predict([x])
    # sample from N(mean, sigma)
    return scipy.stats.norm(mean, sigma).rvs()


def gp_sample_from(
        gp,
        support
):
    """Samples from a gaussian process, gp

    support: sequence of X values to sample at

    returns values at support points
    """
    # raise RuntimeError(
    #       'Not sure this works - see gp_examples.py:fixed_period_example'
    # )
    (mean, sigma, LL) = gp.predict(support)
    # sample from N(mean, sigma)
    sample = numpy.random.multivariate_normal(
        asarray(mean).reshape(len(support), ),
        sigma
    )
    return sample


def gp_plot_samples_from(
        gp,
        support,
        num_samples=1
):
    """
    Plot samples from a Gaussian process.
    """
    from pylab import plot
    mean, sigma, LL = gp.predict(support)
    gp_plot_prediction(support, mean, sigma)
    for i in xrange(num_samples):
        sample = numpy.random.multivariate_normal(
            asarray(mean).reshape(len(support),),
            sigma
        )
        plot([x[0] for x in support], sample)


def gp_plot_prediction(
        predict_x,
        mean,
        variance=None
):
    """
    Plot a gp's prediction using pylab including error bars if variance specified

    Error bars are 2 * standard_deviation as in `Gaussian Processes for Machine Learning`__ by Rasmussen and Williams. 

    __ http://www.amazon.co.uk/Gaussian-Processes-Learning-Adaptive-Computation/dp/026218253X/
    """
    from pylab import plot, concatenate, fill
    if None != variance:
        # check variances are just about +ve - could signify a bug if not
        assert diagonal(variance).all() > -1e-10
        data = [
            (x, y, max(v, 0.0))
            for x, y, v
            in zip(predict_x, mean.flat, diagonal(variance))
        ]
    else:
        data = [
            (x, y)
            for x, y
            in zip(predict_x, mean)
        ]
    data.sort(key=lambda d: d[0])  # sort on X axis
    predict_x = [d[0] for d in data]
    predict_y = asarray([d[1] for d in data])
    plot(predict_x, predict_y, color='k', linestyle=':')
    # if None != variance:
    #     sd = sqrt(asarray([d[2] for d in data]))
    #     var_x = concatenate((predict_x, predict_x[::-1]))
    #     var_y = concatenate(
    #         (predict_y + 2.0 * sd, (predict_y - 2.0 * sd)[::-1]))
    #     p = fill(var_x, var_y, edgecolor='w', facecolor='#d3d3d3')
    #     plot( predict_x, predict_y + 2.0 * sd, 'k--' )
    #     plot( predict_x, predict_y - 2.0 * sd, 'k--' )


def gp_title_and_show(gp):
    """Add title and show using pylab
    """
    from pylab import title, show
    t = 'Log likelihood: %f\n%s' % (gp.LL, gp.k.params)
    print t
    title(t)
    show()


def loo_gen(z, i):
    """Leave one out generator (leaves i'th out)"""
    for j, z_ in enumerate(z):
        if i != j:
            yield z_


def loo(z, i):
    """Returns sequence with i'th missing"""
    return [z_ for z_ in loo_gen(z, i)]


def gp_loo_predict_i(k, X, y, i):
    """Creates a gp from all x in X except the i'th and predicts its value...

    Args:
            k: the kernel for the gp
            X: the data
            y: the outputs
            i: the index to leave out

    Returns:
            ( predicted, variance, error )
            predicted: the i'th data point's predicted value
            variance: uncertainty in prediction
            error: absolute error
    """
    assert len(X) == len(y)
    X_loo = loo(X, i)
    y_loo = asarray(loo(y, i), numpy.float64)
    gp = infpy.gp.GaussianProcess(X_loo, y_loo, k)
    (mean, variance, LL) = gp.predict([X[i]])
    assert mean.shape == (1, 1)
    assert variance.shape == (1, 1)
    predicted = mean[0, 0]
    variance = variance[0, 0]
    error = math.fabs(predicted - y[i])
    return predicted, variance, error


def gp_loo_predict(k, X, y):
    """Calls gp_loo_predict_i for all i and yields the return value"""
    for i in xrange(len(X)):
        yield gp_loo_predict_i(k, X, y, i)


if "__main__" == __name__:
    gp = GaussianProcess(
        [[2.], [1.]],
        [.5, .1],
        SquaredExponentialKernel([1.])
    )
    mean, variance, LL = gp.predict([[.0], [.1], [.2]])
    import IPython
    IPython.Debugger.Pdb().set_trace()
    mean, variance, LL = gp.predict([])
    gp = GaussianProcess(
        [],
        [],
        SquaredExponentialKernel([1.])
    )
    mean, variance, LL = gp.predict([[.0], [.1], [.2]])
    mean, variance, LL = gp.predict([])
