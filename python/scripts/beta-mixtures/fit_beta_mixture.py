#
# Copyright John Reid 2010
#

"""
Fit a mixture of betas to given data.
"""

print 'Start of script'
from time import time
start = time()
import logging, sys, itertools, cPickle
elapsed = time() - start; print 'Importing logging took %f seconds' % elapsed; start = time()
from optparse import OptionParser
elapsed = time() - start; print 'Importing optparse took %f seconds' % elapsed; start = time()
import pylab as pl, numpy as np
elapsed = time() - start; print 'Importing numpy took %f seconds' % elapsed; start = time()
import infpy.mixture.beta; reload(infpy.mixture.beta)
elapsed = time() - start; print 'Importing infpy.mixture.beta took %f seconds' % elapsed; start = time()
from infpy.mixture import beta
elapsed = time() - start; print 'Importing beta took %f seconds' % elapsed; start = time()
from cookbook.script_basics import log_options, setup_logging
elapsed = time() - start; print 'Importing script_basics took %f seconds' % elapsed; start = time()


def plot_pi_bar_prior(alpha):
    filename = 'pi_bar_pdf.png'
    logging.info('Plotting beta(1, %g) prior for pi_bar to %s', alpha, filename)
    import scipy.stats
    domain = np.linspace(0, 1, 1000)[1:-1]
    pdf = scipy.stats.beta.pdf(domain, 1, alpha)
    #print beta.pdf([0, 1], 1, alpha)
    pl.figure()
    pl.semilogy(domain, pdf)
    pl.title('$\\bar{\\pi} \\sim $Beta$(1, %g)$' % alpha)
    pl.xlabel('$\\bar{\\pi}$')
    pl.ylabel('$p(\\bar{\\pi})$')
    pl.savefig(filename)
    pl.close()

def density_estimate(x):
    from scipy.stats import gaussian_kde
    def covariance_factor(self):
        return 0.01
    #setattr(gkde, 'covariance_factor', covariance_factor.__get__(gkde, type(gkde)))
    gaussian_kde.covariance_factor = covariance_factor
    return gaussian_kde(x.reshape(1, len(x)))



def load_data(options):
    "Yield the data points."
    
    def input(filename, num_data, tag):
        if filename:
            logging.info('Reading %s from %s.', tag, filename)
            f = open(filename)
        else:
            logging.info('Reading %s from stdin.', tag)
            if not num_data:
                raise ValueError('Number of data points not specified.')
            f = itertools.islice(sys.stdin, num_data)
        return f
    
    def load_data(f):
        result = []
        for l in f:
            result.extend(map(float, l.split()))
        return result
    
    x = load_data(input(options.x_file, options.N, 'x'))

    num_data = len(x)
    if options.N != 0 and num_data != options.N:
        raise ValueError('Wrong number of data points.')
    
    if not options.weights_file:
        logging.info('Weighting data uniformly.')
        weights = np.ones(num_data)
    else:
        weights = load_data(input(options.weights_file, num_data, 'weights'))
    if len(weights) != num_data:
        raise ValueError('Wrong number of weights.')
    
    return np.asarray(x), np.asarray(weights)



np.seterr(over='warn', invalid='raise')

parser = OptionParser()
beta.add_options(parser)
parser.add_option(
    "-N",
    dest="N",
    type='int',
    help="Number of data points.",
    default=0,
)
parser.add_option(
    "--x-file",
    dest="x_file",
    help="Filename where data are stored (uses stdin if not given).",
    default=None,
)
parser.add_option(
    "--weights-file",
    dest="weights_file",
    help="Filename where weights are stored (uses uniform weights if not given).",
    default=None,
)
parser.add_option(
    "--model-file",
    dest="model_file",
    help="Filename where model is stored and saved (will create new model if file does not exist or option not given).",
    default=None,
)
parser.add_option(
    "--num-starts",
    dest="num_starts",
    help="Number of different starting points to try.",
    type='int',
    default=1,
)
parser.add_option(
    "--plot-file",
    dest="plot_file",
    help="File to plot distribution in.",
    default=None,
)
parser.add_option(
    "--log-plot",
    dest="log_plot",
    help="Use log-scale for plot.",
    action='store_true',
)
parser.add_option(
    "--predictions-file",
    dest="predictions_file",
    help="Filename predictions are written to.",
    default=None,
)
parser.add_option(
    "--seed",
    dest="seed",
    help="Seed for the RNG.",
    type='int',
    default=1,
)
parser.add_option(
    "--log-file",
    dest="log_file",
    help="Log file.",
    type='str',
    default=None,
)
options, args = parser.parse_args()
setup_logging(file=options.log_file, level=logging.INFO)
log_options(parser, options)

exp_family = beta.DirichletExpFamily(k=2)
x, weights = load_data(options)
X = np.empty((len(x), 2))
X[:,0] = x
X[:,1] = 1.-X[:,0]

# get sufficient statistics
T = exp_family.T(X)

# plot prior for pi bar
# plot_pi_bar_prior(options.alpha)

# make reproducible
if options.seed:
    logging.info('Seeding RNG with %d', options.seed)
    np.random.seed(options.seed)

# for each start
mixtures = []
for start in xrange(options.num_starts):
    logging.info('Trying start %s', start+1)

    def create_mixture():
        logging.info('Creating new model.')
        tau = -np.ones(2)
        nu = 1.
#        tau = np.zeros(2)
#        nu = 0.
        return beta.ExpFamilyMixture(T, weights, options.K, exp_family, tau, nu, options=options)
    
    if options.model_file:
        if options.num_starts > 1:
            raise ValueError('No point having more than one start if loading model from file.')
        try:
            mixture = cPickle.load(open(options.model_file))    
            logging.info('Loaded model from %s', options.model_file)
            mixture.set_x(T, weights)
        except:
            mixture = create_mixture()
    else:
        # create a mixture of exponential family distributions
        mixture = create_mixture()
    
    # update mixture
    test = beta.LlConvergenceTest(eps=options.tolerance, should_increase=True, use_absolute_difference=True)
    bound = mixture.variational_bound()
    test(bound)
    for _i in xrange(options.max_iter):
        mixture.update()
        bound = mixture.variational_bound()
        logging.info('Iteration %d: Variational bound = %f', _i+1, bound)
        if test(bound) and _i + 1 >= options.min_iter:
            break
    
    # save to see which has best LL
    mixtures.append((bound, mixture))

# sort by LL and select best
mixtures.sort()
best_bound, best_mixture = mixtures[-1]
logging.info('Best variational bound = %f', best_bound)

# save model if requested
if options.model_file:
    logging.info('Saving model to %s', options.model_file)
    cPickle.dump(best_mixture, open(options.model_file, 'w'))    

# plot distribution if requested    
if options.plot_file:
    logging.info('Plotting distribution to %s', options.plot_file)
    from cookbook.pylab_utils import set_rcParams_for_latex, get_fig_size_for_latex
    set_rcParams_for_latex()
    #fig_size = get_fig_size_for_latex(1000)
    pl.rcParams['figure.figsize'] = (4.652, 3.2)    
    pl.figure()
    mixture_x, mixture = best_mixture.plot(log=options.log_plot, legend=False)
    pl.savefig('mixture-%s.eps' % options.plot_file, format='eps')
    pl.close()
    
    mixture_x, mixture = best_mixture.plot(log=options.log_plot, legend=False, scale=False)
    pl.savefig('mixture-unscaled-%s.eps' % options.plot_file, format='eps')
    pl.close()
    
    beta.plot_density_with_R(x, weights, options.plot_file, mixture_x, mixture, adjust=.15)

# write predictions if requested
if options.predictions_file:
    logging.info('Writing predictions to %s' % options.predictions_file)
    f = open(options.predictions_file, 'w')
    for p in best_mixture.evaluate(T):
        print >> f, p
    f.close()
