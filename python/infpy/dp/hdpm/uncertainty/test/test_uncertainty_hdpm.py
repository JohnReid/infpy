#
# Copyright John Reid 2010
#


"""
Test implementation of HDPM with uncertainty in the words.
"""

import logging
import numpy
import unittest

import infpy.dp.hdpm.math
reload(infpy.dp.hdpm.math)
import infpy.dp.hdpm.uncertainty
reload(infpy.dp.hdpm.uncertainty)
import infpy.dp.hdpm.uncertainty as U
import infpy.dp.hdpm.uncertainty.summarise
reload(infpy.dp.hdpm.uncertainty.summarise)
from infpy.dp.hdpm.uncertainty.summarise import Statistics, InferenceHistory, Summariser


class LLChecker(object):
    def __call__(self, last_LL, LL, tag, raise_errors=True, tolerance=.02):
        total_LL = sum(LL)
        change = sum(LL) - sum(last_LL)
        proportional_change = change / (numpy.abs(total_LL))
        logging.debug('%30s: Log likelihood change: %.2e = %.2f%%',
                      tag, change, 100. * proportional_change)
        if proportional_change < -tolerance:
            logging.warning('LL change=%f. LL=%f. Component changes=\n%s',
                            change, total_LL, str(LL - last_LL))
            logging.warning('LL components=\n%s', str(LL))
            if raise_errors:
                raise RuntimeError(
                    'Log likelihood change=%f. Component changes=\n%s' % (
                        sum(LL) - sum(last_LL),
                        LL - last_LL
                    )
                )
        return LL


def test_log_likelihood_per_update(dist, num_updates=5):
    "Test the log likelihood increases as we do each individual update."
    ll_checker = LLChecker()
    last_LL = numpy.asarray(dist._log_likelihood())
    logging.debug('Initial log likelihood: %f', sum(last_LL))
    for _ in xrange(num_updates):
        for update_fn in U.VariationalDistribution.update_fns:
            update_fn(dist)
            last_LL = ll_checker(last_LL, numpy.asarray(
                dist._log_likelihood()), tag=update_fn.__name__)
            history.update()
    logging.debug('Final log likelihood: %f', sum(last_LL))
    logging.debug('Final log likelihood components:\n%s', str(last_LL))


class TestUncertainty(unittest.TestCase):

    def test_hand_made_data_set(self):
        "Test HDPM with uncertainty on small hand-crafted data set."
        numpy.random.seed(37)
        F = 12
        K = 5
        genes = [
            [
                (0, [.1, .2, .99]),
            ],
            [
                (3, [.9, .9]),
                (2, [.2, .99]),
                (4, [.2, .99]),
                (5, [.2, .99]),
            ],
            [
                (0, [.1, .2, .99]),
                (2, [.1, .2]),
                (10, [.1, .2, .99]),
                (11, [.1, .2]),
                (9, [.1, .2]),
            ],
        ]
        options = U.get_default_options()
        data = U.Data(genes, F, options)
        dist = U.VariationalDistribution(data, K)
        test_log_likelihood_per_update(dist)
        stats = Statistics(dist)
        stats.log()

    def test_sampled_data(self):
        "Test HDPM with uncertainty on sampled data sets of different sizes."
        for i, (F, K, G, average_n_g) in enumerate((
            (2, 10, 20, 5),
            (2, 4, 20, 10),
            (2, 10, 20, 10),
            (2, 10, 220, 50),
            (12, 80, 100, 50),
            (80, 6, 200, 200),
        )):
            numpy.random.seed(i + 1)
            logging.debug(
                'Testing sampled data with F=%d; K=%d; G=%d, average n_g=%d', F, K, G, average_n_g)
            options = U.get_default_options()
            options.a_tau = numpy.ones(F) / F
            options.a_omega = numpy.ones(F) / F
            rho = U.sample_rho(G, average_n_g=average_n_g)
            sample = U.sample(options, rho, K, F)
            genes = U.genes_from_sites(sample.sites, rho)
            data = U.Data(genes, F, options)
            dist = U.VariationalDistribution(data, K)
            test_log_likelihood_per_update(dist)


if '__main__' == __name__:
    logging.basicConfig(level=logging.DEBUG)
    # unittest.main()
    # TestUncertainty('test_hand_made_data_set').debug()
    TestUncertainty('test_sampled_data').debug()
