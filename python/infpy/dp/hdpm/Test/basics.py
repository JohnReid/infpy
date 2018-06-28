#
# Copyright John Reid 2010
#


"""
Define a simple corpus.

The corpus consists of 2 documents over a vocabulary of 2 words. The first document consists entirely of the
first word and the second the second word.
"""

import logging
from infpy.convergence_test import LlConvergenceTest
from optparse import OptionParser


def create_option_parser():
    parser = OptionParser()
    parser.add_option(
        "-s",
        dest="seed",
        default=6,
        type='int',
        help="Seed for RNG."
    )
    parser.add_option(
        "-I",
        dest="max_iters",
        type='int',
        default=150,
        help="Maximum number of iterations."
    )
    parser.add_option(
        "-t",
        dest="LL_tolerance",
        type='float',
        default=0.004,
        help="Tolerance in LL to check for convergence."
    )
    return parser


def parse_options(parser):
    options, args = parser.parse_args()
    logging.info('Options:')
    for option in parser.option_list:
        if option.dest:
            logging.info('%32s: %-32s * %s', option.dest,
                         str(getattr(options, option.dest)), option.help)
    return options, args


def infer_model(model, options):
    "Try to infer one model"

    convergence_test = LlConvergenceTest(
        eps=options.LL_tolerance, should_increase=False, use_absolute_difference=True)
    for iter in xrange(options.max_iters):
        model.update()
        LL = model.log_likelihood()
        logging.debug('Iteration % 2d: LL=%f', iter, LL)
        if convergence_test(LL):
            logging.info(
                'LL converged to %f after %d iterations', LL, iter + 1)
            break
    else:
        logging.warning(
            'Did not converge after %d iterations. LL=%f', iter + 1, LL)
    return LL, model


def summarise_model(model, LL):
    "Log some details about the model."
    logging.info('LL=%f', LL)
    logging.info('Expected # words in each topic:\n%s',
                 str(model.counts.E_n_dk.sum(axis=0)))
    logging.info('Topic per document distributions:\n%s', str(
        (model.counts.E_n_dk.T / model.counts.E_n_dk.sum(axis=1)).T))
    logging.info('Topic distributions:\n%s', str(
        (model.counts.E_n_kw.T / model.counts.E_n_kw.sum(axis=1)).T))
