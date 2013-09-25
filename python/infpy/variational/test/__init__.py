#
# Copyright John Reid 2006, 2010
#

import unittest

def suite():
    import dirichlet_process_test
    suite = unittest.TestSuite()
    suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(dirichlet_process_test))
    return suite
    

if __name__ == "__main__":
    unittest.TextTestRunner(verbosity=2).run(suite())
