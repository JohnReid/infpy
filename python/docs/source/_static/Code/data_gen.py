#
# Copyright John Reid 2012
#

import numpy, pylab, infpy.gp

# Generate some noisy data from a modulated sine curve
x_min, x_max = 10.0, 100.0
X = infpy.gp.gp_1D_X_range(x_min, x_max) # input domain
Y = 10.0 * numpy.sin(X[:,0]) / X[:,0] # noise free output
Y = infpy.gp.gp_zero_mean(Y) # shift so mean=0.
e = 0.03 * numpy.random.normal(size = len(Y)) # noise
f = Y + e # noisy output

# a function to save plots
def save_fig(prefix):
    "Save current figure in extended postscript and PNG formats."
    pylab.savefig('%s.png' % prefix, format='PNG')
    pylab.savefig('%s.eps' % prefix, format='EPS')
  
# plot the noisy data
pylab.figure()
pylab.plot(X[:,0], Y, 'b-', label='Y')
pylab.plot(X[:,0], f, 'rs', label='f')
pylab.legend()
save_fig('simple-example-data')
pylab.close()

