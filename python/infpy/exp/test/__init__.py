#
# Copyright John Reid 2007, 2010
#

import unittest


def suite():
    from . import exp_family_test
    suite = unittest.TestSuite()
    suite.addTests(
        unittest.defaultTestLoader.loadTestsFromModule(exp_family_test))
    return suite


if __name__ == "__main__":
    unittest.TextTestRunner(verbosity=2).run(suite())
