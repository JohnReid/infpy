#
# Copyright John Reid 2007
#

from numpy.random import multivariate_normal
from numpy.linalg import inv
from numpy import outer

mu = [10.,10.]
precision = [[.6, .3], [.3, .6]]
sigma = inv(precision)
sample_size = 10000
num_tests = 100

for test in xrange(num_tests):
    samples = multivariate_normal(mu, sigma, [sample_size])
    sample_mean = samples.sum(axis=0)/sample_size
    #print (sample_mean - mu) > 0.
    sample_covariance = sum(outer(sample-sample_mean, sample-sample_mean) for sample in samples) / (sample_size-1)
    print sample_covariance - sigma
