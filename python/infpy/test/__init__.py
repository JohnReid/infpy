#
# Copyright John Reid 2006, 2010
#

import unittest


def suite():
    import distribution_test
    #import mvn_mixture_test
    import utils_test

    # import infpy.variational.test
    import infpy.decision.test
    # import infpy.dp.hdpm.uncertainty.test
    import infpy.gp.test
    import infpy.exp.test
    import infpy.mixture.test

    suite = unittest.TestSuite()

    # suite.addTests(infpy.variational.test.suite())
    suite.addTests(infpy.decision.test.suite())
    # suite.addTests(infpy.dp.hdpm.uncertainty.test.suite())
    suite.addTests(infpy.gp.test.suite())
    suite.addTests(infpy.exp.test.suite())
    suite.addTests(infpy.mixture.test.suite())

    suite.addTests(
        unittest.defaultTestLoader.loadTestsFromModule(distribution_test))
    # suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(mvn_mixture_test))
    suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(utils_test))

    return suite


if '__main__' == __name__:
    unittest.TextTestRunner(verbosity=2).run(suite())
