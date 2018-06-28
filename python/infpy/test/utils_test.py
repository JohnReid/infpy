#
# Copyright John Reid 2006
#


from infpy import *
import unittest


class CloseToTest(unittest.TestCase):
    def test(self):
        for f1, f2, close_to in [
                (0.1, 0.10000001, True),
                (0.1, 0.1000001, True),
                (0.1, 0.100001, False)
        ]:
            assert check_is_close(f1, f2) == close_to, \
                "%f %f should %s be close" % (f1, f2, close_to and "" or "not")


class ZeroMeanUnityVarianceTest(unittest.TestCase):
    def test(self):
        Y = [
            numpy.random.uniform(size=4)
            for i in xrange(5)
        ]
        Scaled = []
        Revert = []
        for y in Y:
            scaled, revert = zero_mean_unity_variance(y)
            Scaled.append(scaled)
            Revert.append(revert)
        for y, scaled, revert in zip(Y, Scaled, Revert):
            n = norm2(y - revert(scaled))
            assert n < 1e-10
            assert math.fabs(numpy.mean(scaled)) < 1e-8
            assert math.fabs(numpy.std(scaled) - 1.0) < 1e-8


if __name__ == "__main__":
    unittest.main()
