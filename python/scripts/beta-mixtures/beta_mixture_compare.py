#
# Copyright John Reid 2011
#

"""
Compare different models for beta mixtures.
"""

import logging
from optparse import OptionParser
import numpy as np
from cookbook.script_basics import log_options, setup_logging
from IPython.kernel import client
from collections import defaultdict, namedtuple
import infpy.mixture.beta; reload(infpy.mixture.beta)
from infpy.mixture import beta
from infpy.utils import k_fold_cross_validation
from cookbook.pylab_utils import violin_plot
import pylab as pl

TrainTask = namedtuple('TrainTask', ('id', 'mixture', 'validation'))

def density_estimate(x):
    from scipy.stats import gaussian_kde
    def covariance_factor(self):
        return 0.01
    #setattr(gkde, 'covariance_factor', covariance_factor.__get__(gkde, type(gkde)))
    gaussian_kde.covariance_factor = covariance_factor
    return gaussian_kde(x.reshape(1, len(x)))



def load_data(filename):
    "Yield the data points."    
    x = []
    for l in open(filename):
        x.extend(map(float, l.split()))
    return np.asarray(x)



def create_mixture(T):
    logging.debug('Creating new model.')
    tau = -np.ones(2)
    nu = 1.
#    tau = np.zeros(2)
#    nu = 0.
    mixture = beta.ExpFamilyMixture(T, np.ones(len(T)), options.K, exp_family, tau=tau, nu=nu, options=options)
    return mixture


def train_model(mixture, validation, options):
    import infpy.mixture.beta; reload(infpy.mixture.beta)
    from infpy.mixture import beta
    import logging, numpy as np
    test = beta.LlConvergenceTest(eps=options.tolerance, should_increase=True, use_absolute_difference=True)
    test(mixture.variational_bound())
    for _i in xrange(options.max_iter):
        mixture.update()
        bound = mixture.variational_bound()
        logging.debug('Iteration %d: Variational bound = %f', _i+1, bound)
        if test(bound) and _i + 1 >= options.min_iter:
            break
    return bound, np.log(mixture.evaluate(np.asarray(validation)))
    

def train_mixtures(task_ids, fold, training, validation):
    # make reproducible
    if options.seed:
        np.random.seed(options.seed)
        
    # for each start
    for start in xrange(options.num_starts):
        logging.debug('Trying start %s', start+1)
        mixture = create_mixture(np.asarray(training))
        # pass to parallelism
        task_ids[fold][start] = TrainTask(
            id=tc.run(
                client.MapTask(
                    train_model,
                    (mixture, validation, options)
                ),
                block=False
            ),
            mixture=mixture, 
            validation=validation
        )


def start_cross_validation(task_ids, T):
    for fold, (training, validation) in enumerate(k_fold_cross_validation(T, options.x_validate_groups)):
        train_mixtures(task_ids, fold, training, validation)


def gather_results(task_ids):
    LLs = []
    for fold, starts in task_ids.iteritems():
        results = [tc.get_task_result(task.id, block=True) for start, task in starts.iteritems()]
        results.sort()
        _best_bound, best_LLs = results[-1]
        LLs.extend(best_LLs)
    return LLs


def kde_cross_validate(x, adjust=0.05):
    import rpy2.robjects as robjects
    from rpy2.robjects import r as R
    from rpy2.robjects.packages import importr
    log_predictions = []
    for training, validation in k_fold_cross_validation(x, options.x_validate_groups):
        r_x = robjects.FloatVector(training)
        density_args = {
            'from' : 0,
            'to' : 1,
            'adjust' : adjust,
        }
        kde = R['density'](r_x, **density_args)
        predictions = R['approx'](kde, xout=robjects.FloatVector(validation))
        log_predictions.extend(np.log(predictions[1]))
    return log_predictions


def inv_sigmoid(x):
    x = np.asarray(x)
    return (np.log(x / (1.-x)))
    
    
def mog_cross_validate(x):
    import rpy2.robjects as robjects
    from rpy2.robjects import r as R
    from rpy2.robjects.packages import importr
    R.library('mclust')
    log_predictions = []
    for training, validation in k_fold_cross_validation(x, options.x_validate_groups):
        r_x = robjects.FloatVector(inv_sigmoid(training))
        clustering = R.Mclust(r_x)
        predictions = R.dens(
            modelName = clustering.rx2('modelName'),
            data = robjects.FloatVector(validation),
            parameters = clustering.rx2('parameters')
        )
        log_predictions.extend(np.log(predictions))
    return log_predictions

def append_result(results, text, LLs):
    text = '%s\n%.1f' % (text, sum(LLs))
    logging.info(text)
    results.append((text, LLs))
    
    
def main(filename):
    # load p-values
    x = load_data(filename)
    
    # convert to correct form for exponential family
    X = np.empty((len(x), 2)) 
    X[:,0] = x
    X[:,1] = 1.-X[:,0]
    
    # convert to sufficient statistics
    T = exp_family.T(X)
    
    # plot kernel density estimates of folds
#    for fold, (training, validation) in enumerate(k_fold_cross_validation(T, options.x_validate_groups)):
#        fold_density_filename = 'fold-%02d-density.eps' % fold
#        logging.info('Plotting kernel density estimate for fold %d to %s', fold, fold_density_filename)
#        beta.plot_density_with_R(np.exp([t[0] for t in training]), np.ones(len(training)), fold_density_filename, adjust=.2)

    results = []

    for adjust in (.15, .25, .5):
        append_result(results, 'Kernel\ndensity\n%.3f' % adjust, kde_cross_validate(x, adjust))

    append_result(results, 'Mixture\nof\nGaussians', mog_cross_validate(x))

    for point_estimates in (True, ):
        options.point_estimates = point_estimates
        logging.debug('Point estimates: %s' % options.point_estimates)
        
        for stick_breaking in (True, False):
            options.stick_breaking = stick_breaking
            logging.debug('Stick breaking: %s' % options.stick_breaking)
            
            for integrate_pi in (True,):
                options.integrate_pi = integrate_pi
                logging.debug('Integrate pi: %s' % options.integrate_pi)
                
                task_ids = defaultdict(dict)
                start_cross_validation(task_ids, T)
                LLs = gather_results(task_ids)
                append_result(
                    results,
                    "%s\n%s\n%s" % (
                        options.point_estimates and "Point" or "Full",
                        options.stick_breaking and "Stick" or "Dirichlet",
                        options.integrate_pi and "Integrate" or "Infer"
                    ),
                    LLs
                )
    
    return results


def plot_results(results):
    # Make plot to compare different methods   
    old_font_size = pl.rcParams['font.size']
    pl.rcParams['font.size'] = 8
    pl.rcParams['figure.subplot.bottom'] = .3
    pl.figure(figsize=(4.652, 3.2))
    positions = range(len(results))
    violin_plot(pl.gca(), [LLs for title, LLs in results], positions)
    pl.gca().xaxis.set_ticks(positions)
    pl.gca().xaxis.set_ticklabels([title for title, LLs in results])
    pl.ylabel('Predictive log likelihood of held-out test data')
    pl.savefig('method-comparison.eps')
    pl.clf()
    pl.rcParams['font.size'] = old_font_size
    


np.seterr(over='warn', invalid='raise')

parser = OptionParser()
beta.add_options(parser)
parser.add_option(
    "--num-starts",
    dest="num_starts",
    help="Number of different starting points to try.",
    type='int',
    default=1,
)
parser.add_option(
    "--x-validate-groups",
    help="Number of cross-validation groups.",
    type='int',
    default=5,
)
parser.add_option(
    "--seed",
    dest="seed",
    help="Seed for the RNG.",
    type='int',
    default=1,
)
options, args = parser.parse_args()
if 1 != len(args):
    raise ValueError('Need to specify data file.')
filename = args[0]
setup_logging(level=logging.INFO)
log_options(parser, options)

exp_family = beta.DirichletExpFamily(k=2)
    
# create object for parallelism
tc = client.TaskClient()

results = main(filename)
plot_results(results)
