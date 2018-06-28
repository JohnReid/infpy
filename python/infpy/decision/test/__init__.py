#
# Copyright John Reid 2007, 2010
#

import unittest


def suite():
    from . import decision_test
    from . import genepy_test
    from . import rule_test
    suite = unittest.TestSuite()
    suite.addTests(
        unittest.defaultTestLoader.loadTestsFromModule(decision_test))
    suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(genepy_test))
    suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(rule_test))
    return suite


if __name__ == "__main__":
    unittest.TextTestRunner(verbosity=2).run(suite())
