#
# Copyright John Reid 2010
#

"""
Code to examine if a beta mixture has bad local minima.
"""

import numpy as np, scipy as sc, logging
from scipy.special import betaln
from scipy.optimize import fmin_cg

logging.basicConfig(level=logging.INFO)

np.seterr(over='warn')
np.random.seed(1)

N = 40
x = np.empty(N)
"Data."
x[:N/2] = np.random.beta(3, 1, N/2)
x[N/2:] = np.random.beta(1, 3, N/2)

T = np.empty((N, 2))
"Sufficient statistics of data."
T[:,0] = np.log(x)
T[:,1] = np.log(1-x)


K = 2
"Number of components."

def logistic(x):
    "The logistic function."
    return 1. / (1. + np.exp(-x))

def logistic_inv(y):
    "The inverse of the logistic function."
    return np.log(y / (1. - y))

def encode_multinomial_params(w):
    "Encode a distribution summing to 1."
    return logistic_inv(np.rollaxis(w, -1)[:-1])

def decode_multinomial_params(params):
    "Encode a distribution summing to 1."
    shape = list(params.shape)
    shape[-1] += 1
    w = np.empty(shape)
    w_part = np.rollaxis(np.rollaxis(w, -1)[:-1], -1)
    w_part[:] = logistic(params)
    np.rollaxis(w, -1)[-1] = 1. - w_part.sum(axis=-1)
    assert (w >= 0.).all()
    assert (w <= 1.).all()
    assert (.99 < w.sum(axis=-1)).all()
    assert (w.sum(axis=-1) < 1.01).all()
    return w

def A(eta):
    "Log partition function."
    return betaln(np.rollaxis(eta, -1)[0] + 1., np.rollaxis(eta, -1)[1] + 1.)

def log_likelihood(w, eta):
    "Log likelihood function."
    assert (N, K) == w.shape
    assert (K, 2) == eta.shape
    return (w * (np.dot(eta, T.T).T - A(eta))).sum()

def to_minimise(params):
    "Function to minimise."
    assert (N*(K-1) + 2*K,) == params.shape
    return -log_likelihood(*decode_params(params))

def decode_params(params):
    "Decode flattened parameters to (w, eta)."
    return decode_multinomial_params(params[:N*(K-1)].reshape((N, K-1))), np.expm1(params[N*(K-1):].reshape((K, 2)))

def encode_params(w, eta):
    "Encode eta and w into flattened parameters."
    params = np.empty(N*(K-1) + 2*K)
    params[:N*(K-1)] = encode_multinomial_params(w).reshape(N*(K-1))
    params[N*(K-1):] = np.log1p(eta).reshape(2*K)
    return params

for seed in xrange(1,21):
    
    # Seeds 3 and 25 seem to result in inferior local optima
    logging.info('Seeding numpy.random with %d', seed)
    np.random.seed(seed)
    
    #starting_eta = np.array([
    #    (-.99, -.98),
    #    (-.98, -.99),
    #])
    starting_eta = np.random.exponential(scale=1., size=(K, 2)) - 1.
    "Initial eta for optimisation."
    
    starting_w = np.ones((N, K)) / K
    "Initial w for optimisation."
    
    initial_params = encode_params(starting_w, starting_eta)
    "Initial parameters for optimisation."
    
    # run optimisation
    xopt, fopt, func_calls, grad_calls, warnflag = fmin_cg(to_minimise, initial_params, full_output=True, disp=False)
    if 1 & warnflag:
        logging.warning('Maximum number of iterations exceeded.')
        raise RuntimeError('Maximum number of iterations exceeded.')
    elif 2 & warnflag:
        logging.warning('Gradient and/or function calls not changing.')
        #raise RuntimeError('Gradient and/or function calls not changing.')
        continue
    logging.info('Optimisation terminated succesfully: expected LL=%.4e; # func calls=%d; # grad calls=%d', -fopt, func_calls, grad_calls)
    ml_w, ml_eta = decode_params(xopt)
    if -fopt < 1.4e12 and -fopt > -np.infty:
        logging.info('Starting eta:\n%s', str(starting_eta))
        logging.info('ML eta:\n%s', str(ml_eta))
    
