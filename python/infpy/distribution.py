#
# Copyright John Reid 2006
#


import math
import numpy.random


class Distribution(object):
    """Base class for distributions"""

    def supports(self, x):
        """Returns if x is in the distribution's support"""
        return True

    def plot(self, start, end, num_steps=100):
        import pylab
        import numpy
        start = float(start)
        end = float(end)
        num_steps = float(num_steps)
        x = numpy.arange(start, end, (end - start) / num_steps)
        y = [math.exp(self.log_pdf(x1)) for x1 in x]
        pylab.plot(x, y)

    def mean(self):
        """Mean of the distribution"""
        raise RuntimeError("Needs to be implemented in base class")

    def variance(self):
        """Variance of the distribution"""
        raise RuntimeError("Needs to be implemented in base class")

    def sample_from(self):
        """Samples one value from the distribution"""
        raise RuntimeError("Needs to be implemented in base class")

    def __str__(self):
        """Returns a string representation of the distribution"""
        raise RuntimeError("Needs to be implemented in base class")


class NormalDistribution(Distribution):
    """http://en.wikipedia.org/wiki/Normal_distribution"""

    def __init__(self, mu=1.0, sigma=1.0):
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.c = 1.0 / (self.sigma * math.sqrt(2 * math.pi))
        self.log_c = math.log(self.c)

    def log_pdf(self, x):
        return (
            self.log_c
            - (((x - self.mu) / self.sigma) ** 2) / 2
        )

    def dlog_pdf_dx(self, x):
        return (self.mu - x) / (self.sigma ** 2)

    def mean(self):
        """Mean of the distribution"""
        return self.mu

    def variance(self):
        """Variance of the distribution"""
        return self.sigma

    def sample_from(self):
        """Samples one value from the distribution"""
        return numpy.random.normal(self.mu, math.sqrt(self.sigma))

    def __str__(self):
        """Returns a string representation of the distribution"""
        return "Normal( mu=%f, sigma=%f )" % (self.mu, self.sigma)


class LogNormalDistribution(Distribution):
    """http://en.wikipedia.org/wiki/Log_normal_distribution"""

    def __init__(self, mu=1.0, sigma=1.0):
        """Remember mu and sigma are mean and stdev of distribution's logarithm"""
        self.mu = float(mu)
        self.sigma = float(sigma)

    def supports(self, x):
        """Returns if x is in the distribution's support"""
        return 0.0 < x

    def log_pdf(self, x):
        if 0.0 >= x:
            return math.log(1e-300)
        log_x = math.log(x)
        return (
            - log_x
            - math.log(self.sigma)
            - 0.5 * math.log(2 * math.pi)
            - ((log_x - self.mu) ** 2) / (2 * (self.sigma ** 2))
        )

    def dlog_pdf_dx(self, x):
        if 0.0 >= x:
            return 0.0
        log_x = math.log(x)
        sigma_sq = self.sigma ** 2
        result = - (log_x + sigma_sq - self.mu) / (x * sigma_sq)
        # print x, log_x, result
        return result

    def mean(self):
        """Mean of the distribution"""
        return math.exp(self.mu + (self.sigma ** 2) / 2)

    def variance(self):
        """Variance of the distribution"""
        return (math.exp(self.sigma ** 2) - 1.0) \
            * math.exp(2.0 * self.mu + self.sigma ** 2)

    def sample_from(self):
        """Samples one value from the distribution"""
        return numpy.random.lognormal(mean=self.mu, sigma=self.sigma)

    def __str__(self):
        """Returns a string representation of the distribution"""
        return "LogNormal( mu=%f, sigma=%f )" % (self.mu, self.sigma)


class GammaDistribution(Distribution):
    """http://en.wikipedia.org/wiki/Gamma_distribution

    Mean is k * theta
    Support is [0,inf)
    """

    def __init__(self, k=1.0, theta=1.0):
        self.k = float(k)
        self.theta = float(theta)

    def supports(self, x):
        """Returns if x is in the distribution's support"""
        return 0.0 < x

    def log_pdf(self, x):
        import scipy.special
        if 0.0 >= x:
            return math.log(1e-300)
        log_x = math.log(x)
        k = self.k
        theta = self.theta
        return (
            (k - 1.0) * log_x
            - (x / theta)
            - k * math.log(theta)
            - math.log(scipy.special.gamma(k))
        )

    def dlog_pdf_dx(self, x):
        if 0.0 >= x:
            return 0.0
        k = self.k
        theta = self.theta
        return (
            (k - 1.0) / x
            - (1.0 / theta)
        )

    def mean(self):
        """Mean of the distribution"""
        return self.k * self.theta

    def variance(self):
        """Variance of the distribution"""
        return self.k * self.theta ** 2

    def sample_from(self):
        """Samples one value from the distribution"""
        return numpy.random.gamma(self.k, self.theta)

    def __str__(self):
        """Returns a string representation of the distribution"""
        return "Gamma( k=%f, theta=%f )" % (self.k, self.theta)


def plot_distribution(d, start=0.01, stop=10.0, resolution=0.1):
    """Displays a plot of the pdf of d"""
    import pylab
    X = numpy.arange(start, stop, resolution)
    Y = [math.exp(d.log_pdf(x)) for x in X]
    pylab.plot(X, Y)


if __name__ == '__main__':
    from pylab import figure, plot, show
    figure
    for sigma in [.125, .25, .5, 1, 1.5, 10]:
        dist = LogNormalDistribution(0, sigma)
        x = []
        y = []
        for i in range(300):
            x.append(float(i + 1) / 100)
            y.append(math.exp(dist.log_pdf(float(i + 1) / 100)))
        plot(x, y)
    show()
