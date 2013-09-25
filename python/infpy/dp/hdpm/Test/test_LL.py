#
# Copyright John Reid 2010
#


"""
Code to test the log likelihood of a HDPM increases as we learn.
"""

from simple_corpus import *

logging.basicConfig(level=logging.INFO)


parser = create_option_parser()
parser.add_option(
    "-n",
    dest="num_models_to_test",
    default=1,
    type='int',
    help="Number of models to test."
)
parser.add_option(
    "--throw",
    dest="throw_exception_on_LL_rise",
    default=False,
    action='store_true',
    help="Throw an exception whenever the LL rises."
)
parser.add_option(
    "-v",
    dest="verbose",
    default=False,
    action='store_true',
    help="Verbose output."
)
parser.add_option(
    "-e",
    dest="increase_eps",
    type='float',
    default=.1,
    help="Threshold for decrease in LL."
)
options, args = parse_options(parser)
numpy.random.seed(options.seed)


class LLchecker(object):
    "Checks LL behaves nicely (always improves)."

    def __init__(self, name, start_LL=None):
        self.name = name
        self.last_LL = start_LL
        self.history = []
        if None != start_LL:
            self.history.append(start_LL)

    def __call__(self, LL, tag, iter):
        self.history.append(LL)
        passed_test = True
        if self.last_LL:
            LL_diff = LL - self.last_LL
            passed_test = LL_diff > -options.increase_eps
            if not passed_test:
                message = '%10s; Iteration: %2d; LL got worse by %2.3f after updating %s.' % (
                    self.name, iter, -LL_diff, tag
                )
                logging.warning(message)
                if options.throw_exception_on_LL_rise:
                    raise RuntimeError(message)
        self.last_LL = LL
        return passed_test


def log_likelihood(model):
    "@return: Terms in the variational bound on the log likelihood as a numpy.array."
    return numpy.array(model._log_likelihood())



#
# Learn some models and check LL
#
for m in xrange(options.num_models_to_test):
    logging.info('Learning model %d.', m)
    model = hdpm.HDPM(documents, W, K)

    # burn in for some iterations
    #model.update()
    #model.update()

    # start checking the LL
    LL = model.log_likelihood()
    checker = LLchecker('FINE', LL)
    coarse_checker = LLchecker('COARSE', LL)
    convergence_test = LlConvergenceTest(eps=.01, should_increase=False, use_absolute_difference=True)
    for iter in xrange(options.max_iters):

        for update_fn in hdpm.HDPM.update_fns:
            before = log_likelihood(model)
            update_fn(model)
            after = log_likelihood(model)
            if not checker(log_likelihood(model).sum(), update_fn.__name__, iter) and options.verbose:
                #logging.warning('LL before: %s', str(before))
                #logging.warning('LL after : %s', str(after))
                logging.warning('LL diff  : %s', str(after - before))

        coarse_checker(log_likelihood(model).sum(), 'iteration', iter)

        if convergence_test(model.log_likelihood()):
            break

    logging.info('Ran % 3d iterations to get to LL=%f', iter+1, model.log_likelihood())
