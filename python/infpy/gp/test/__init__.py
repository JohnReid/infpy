#
# Copyright John Reid 2006, 2010
#

import unittest


def suite():
    import gp_test
    import kernel_test
    suite = unittest.TestSuite()
    suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(gp_test))
    suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(kernel_test))
    return suite


if __name__ == "__main__":
    unittest.TextTestRunner(verbosity=2).run(suite())
