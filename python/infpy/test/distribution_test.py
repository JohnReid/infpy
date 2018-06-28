#
# Copyright John Reid 2006
#


import unittest
import infpy
import math
import numpy


def checked_sample(dist):
    """Checks the sample is in the support of the distribution"""
    x = dist.sample_from()

    # must be in support of distribution
    if not dist.supports(x):
        raise RuntimeError("""Sampling must generate values (%d) in """
                           """support of distribution (%s)""" % (x, str(type(dist))))

    return x


class DistributionTest(unittest.TestCase):
    """Test case for distributions"""

    def test_str(self):
        """Test of string representation of distribution"""
        [str(dist_gen()) for dist_gen in infpy.Distribution.__subclasses__()]

    def test_sample(self):
        """Test of sample from distribution - also tests mean and variance"""
        for dist_gen in infpy.Distribution.__subclasses__():
            try:
                # create the distribution
                dist = dist_gen()

                # create some samples
                num_samples = 100000
                samples = numpy.zeros(num_samples, numpy.float64)
                for i in range(num_samples):
                    samples[i] = checked_sample(dist)

                # check statistics
                empirical_mean = numpy.mean(samples)
                theoretical_mean = dist.mean()
                theoretical_stdev = math.sqrt(dist.variance() / num_samples)
                if(
                        math.fabs(empirical_mean - theoretical_mean)
                        > 3.0 * theoretical_stdev
                ):
                    raise RuntimeError("""%s: Sample mean (%f) not within three """
                                       """standard deviation (%f) of theoretical mean (%f). """
                                       """# samples = %d"""
                                       % (
                                           str(dist),
                                           empirical_mean,
                                           theoretical_stdev,
                                           theoretical_mean,
                                           num_samples
                                       ))

            except Exception as detail:
                print("""Problem with distribution %s: %s""" % (
                    str(dist), detail))
                raise

    def test_pdf_derivative(self):
        """Test of the derivative of the pdf"""
        for dist_gen in infpy.Distribution.__subclasses__():
            try:
                # create the distribution
                dist = dist_gen()

                # try a few samples
                for i in range(30):
                    x = checked_sample(dist)

                    # check the gradients of the pdf
                    infpy.check_gradients(
                        lambda x: dist.log_pdf(x[0]),
                        lambda x: dist.dlog_pdf_dx(x[0]),
                        [x])

            except Exception as detail:
                raise RuntimeError("""Problem with distribution %s: %s"""
                                   % (str(dist), detail))


def show_samples(dist):
    """Test of the distribution's sampling method"""
    import pylab
    # try a few samples
    num_samples = 10000
    samples = numpy.zeros(num_samples, numpy.float64)
    for i in range(num_samples):
        samples[i] = checked_sample(dist)
    pylab.figure()
    pylab.hist(samples, bins=100)
    pylab.title(str(dist))
    pylab.show()


if __name__ == "__main__":
    # [
    #       show_samples( dist_gen() )
    #       for dist_gen
    #       in infpy.Distribution.__subclasses__()
    # ]
    # show_samples( infpy.LogNormalDistribution( 0.0, .25 ) )

    unittest.main()
