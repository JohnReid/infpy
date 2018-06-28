#
# Copyright John Reid 2006
#

import scipy
import numpy
import math


def index_filter(predicate, iterable):
    for i, x in enumerate(iterable):
        if predicate(i, x):
            yield x


def k_fold_cross_validation(X, K, randomise=False):
    """
    Generates K (training, validation) pairs from the items in X.

    The validation iterables are a partition of X, and each validation
    iterable is of length len(X)/K. Each training iterable is the
    complement (within X) of the validation iterable, and so each training
    iterable is of length (K-1)*len(X)/K.

    For example::

        X = [i for i in xrange(97)]
        for training, validation in k_fold_cross_validation(X, K=7):
            for x in X: assert (x in training) ^ (x in validation), x
    """
    if K < 2:
        raise ValueError('Must use at least 2 cross-validation groups.')
    if randomise:
        from random import shuffle
        X = list(X)
        shuffle(X)
    for k in xrange(K):
        training = [x for i, x in enumerate(X) if i % K != k]
        validation = [x for i, x in enumerate(X) if i % K == k]
        yield training, validation


def lu_inv(L):
    from numpy.linalg import solve
    from numpy import eye
    n = L.shape[0]
    K_inv = solve(L.T, solve(L, eye(n)))
    return K_inv


def matrix_from_function(f, shape, dtype, symmetric=False):
    from numpy import zeros, matrix
    result = zeros(shape, dtype)
    for i in range(shape[0]):
        for j in range(symmetric and i + 1 or shape[1]):
            result[i, j] = f(i, j)
            if symmetric and i != j:
                result[j, i] = result[i, j]
    return matrix(result)


def zero_mean_unity_variance(y):
    """Scales the data to make the variance 1 and the mean 0

    Returns (scaled, revert) where
            scaled: the scaled and shifted data
            revert: a unary function that converts back to original data
    """
    m = numpy.mean(y)
    std = numpy.std(y)

    def revert(y_prime):
        return y_prime * std + m

    return (y - m) / std, revert


def norm2(x):
    """Calculates the L2 norm of the vector"""
    return math.sqrt(numpy.sum((x * x) / len(x)))


def calc_grad_approx_differences(f, dfdx, x0):
    from scipy.optimize import approx_fprime
    from scipy.optimize.optimize import _epsilon
    return dfdx(x0) - approx_fprime(x0, f, _epsilon)


def check_is_close_2(left, right, tol=1e-4, strong_or_weak=True):
    import math
    if left == right:
        return True
    diff = math.fabs(left - right)
    try:
        d1 = diff / math.fabs(right)
    except ZeroDivisionError:
        d1 = tol + 1
    try:
        d2 = diff / abs(left)
    except ZeroDivisionError:
        d2 = tol + 1
    if strong_or_weak:
        return d1 <= tol and d2 <= tol
    else:
        return d1 <= tol or d2 <= tol


def check_is_close(f1, f2, tol=1e-6, strong_check=True):
    import math
    abs1 = math.fabs(f1)
    abs2 = math.fabs(f2)
    abs_diff = math.fabs(f1 - f2)
    check1 = 0.0 == abs1 or abs_diff / abs1 <= tol
    check2 = 0.0 == abs2 or abs_diff / abs2 <= tol
    if strong_check:
        return check1 and check2
    else:
        return check1 or check2


def matrix_is_close(A, B, eps=1e-3):
    """Simple test that A and B differ by at most eps in any position"""
    if A.shape != B.shape:
        raise RuntimeError(
            'A and B must have same shape: %s, %s'
            % (str(A.shape), str(B.shape))
        )
    for i in xrange(A.shape[0]):
        for j in xrange(A.shape[1]):
            if not check_is_close(A[i, j], B[i, j], tol=eps):
                return False
    return True


def array_is_close(A, B, eps=1e-3):
    """Checks if two numpy arrays are close"""
    return (
        numpy.fabs(A - B).max()
        / max(numpy.fabs(A).max(), numpy.fabs(B).max())
        < eps
    )


def check_matrix_is_close(A, B, message, eps=1e-3):
    """Raises error and prints message if matrices are not close"""
    if not matrix_is_close(A, B, eps):
        print '%s:\n%s\nand\n%s\ndiffer by:\n%s' % (
            message,
            str(A),
            str(B),
            str(A - B)
        )
        raise RuntimeError('%s' % message)


def check_gradients(f, fprime, x, eps=1e-4):
    """Check the approximation to the gradient of the function matches
    the supplied gradient

    f is a function
    fprime is a function describing the gradient of f

    The gradient will be approximated by expansion of f around x and
    compared with the value of fprime at x
    """
    from scipy.optimize import approx_fprime
    from scipy.optimize.optimize import _epsilon
    calculated = fprime(x)
    approximation = approx_fprime(x, f, _epsilon)
    diff = calculated - approximation
    norm = scipy.sqrt(numpy.dot(diff, diff))
    if norm > eps:
        raise RuntimeError(
            (
                'Gradient does not match approximation from function\n'
                + 'Difference (norm): %f\n'
                + 'x: %s\n'
                + 'Approximation: %s\n'
                + 'Calculated: %s\n'
                + 'differences: %s'
            ) % (
                norm,
                x,
                calculated,
                approximation,
                diff
            ))


def plot_line(start, end, *arguments, **keywords):
    """Plot a line from start to end"""
    import pylab
    pylab.plot(
        [start[0], end[0]],
        [start[1], end[1]],
        *arguments,
        **keywords
    )


def plot_gaussian(mu, sigma, *args, **kwds):
    """Plot a gaussian with given mu and sigma (first 2 dimensions only)
    """
    import pylab
    import numpy
    mu = numpy.asarray(mu).reshape((2,))
    sigma = numpy.asarray(sigma[0:2, 0:2]).reshape((2, 2))
    (u, v) = numpy.linalg.eig(sigma)
    pylab.plot([mu[0]], [mu[1]], 'rs', *args, **kwds)
    plot_line(mu - u[0] * v[:, 0], mu + u[0] * v[:, 0], 'kx--', *args, **kwds)
    plot_line(mu - u[1] * v[:, 1], mu + u[1] * v[:, 1], 'kx--', *args, **kwds)
    # pylab.plot( mu - v[:,1], mu + v[:,1] )


def plot_gaussian_test():
    import pylab
    import numpy
    plot_gaussian([0, 0], numpy.eye(2))
    plot_gaussian([0.5, -0.5], 0.1 * numpy.array([[1, .3], [.3, 1]]))
    pylab.show()
