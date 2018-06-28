#
# Copyright John Reid 2008
#

"""
Code to implement convergence tests (primarily for sequences of log
likelihoods).
"""

import logging


def check_LL_increased(last_LL, LL, tag="", tolerance=.01, raise_error=True):
    """
    Takes 2 numpy arrays of components of a log likelihood. Assumes the total
    LL is the sum of each array. Compares the 2 LLs and if the new one is smaller than
    the first by at least tolerance, raises an error.
    """
    LL_sum, last_LL_sum = LL.sum(), last_LL.sum()
    abs_change = LL_sum - last_LL_sum
    if abs_change < -tolerance:
        msg = '%s: LL has decreased %f from %f to %f:\nFrom:%s\nTo:  %s\nDiff:%s' % (
            tag, LL_sum - last_LL_sum, last_LL_sum, LL_sum, last_LL, LL, LL - last_LL
        )
        logging.warning(msg)
        if raise_error:
            raise ValueError(msg)
    return LL


class LlConvergenceTest(object):
    """Tests for convergence of a series of log likelihoods."""

    def __init__(self, eps=1e-8, should_increase=True, use_absolute_difference=False):
        """
        @arg eps: The tolerance for the convergence test.
        @arg should_increase: If true a warning is printed if the log likelihoods don't always increase.
        @arg use_absolute_difference: If true the absolute differences is used for the convergence test, otherwise any decrease is
        viewed as convergence.
        """
        self.LLs = list()
        "The log likelihoods."

        self.should_increase = should_increase
        "If true a warning is printed if the log likelihoods don't always increase."

        self.eps = eps
        "If true a warning is printed if the log likelihoods don't always increase."

        self.use_absolute_difference = use_absolute_difference
        "If true the absolute differences is used for the convergence test, otherwise any decrease is viewed as convergence."

    def __call__(self, LL):
        "@return: True iff converged."
        self.LLs.append(LL)
        if len(self.LLs) < 2:
            return False
        else:
            if self.should_increase:
                if self.LLs[-1] < self.LLs[-2]:
                    logging.warning(
                        'Iteration %4d: Log likelihoods are not increasing as expected: %f < %f'
                        % (len(self.LLs), self.LLs[-1], self.LLs[-2])
                    )
                    # raise RuntimeError('Log likelihoods are not increasing as expected: %f < %f' % (self.LLs[-1], self.LLs[-2]))
            diff = self.LLs[-1] - self.LLs[-2]
            if self.use_absolute_difference:
                diff = abs(diff)
            return diff < self.eps
