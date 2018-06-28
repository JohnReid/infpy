#
# Copyright John Reid 2010
#

"""
Code to test mixtures.
"""

import unittest
from . import beta_test


def suite():
    suite = unittest.TestSuite()
    suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(beta_test))
    return suite
