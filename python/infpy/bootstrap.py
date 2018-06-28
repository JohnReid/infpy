#
# Copyright John Reid 2009
#


"""
Code to handle bootstrap analyses.
"""

from itertools import cycle
import random
import bisect


def generate_bootstrap_samples(num_samples, test_universe, test_set_sizes):
    """
    Yield samples that match the sizes given in test_set_sizes
    """
    for sample_idx, sample_size in zip(xrange(num_samples), cycle(test_set_sizes)):
        yield random.sample(test_universe, sample_size)


def calculate_bootstrap_statistics(samples, statistic):
    "Calculate the bootstrap statistics for the samples."
    stats = map(statistic, samples)
    stats.sort()
    return stats


def bootstrap_p_value(bootstrap_stats, stat_value):
    """
    Calculate the p-value for the statistic's value given the bootstrap values.
    """
    return 1. - bisect.bisect_left(bootstrap_stats, stat_value) / float(len(bootstrap_stats))
