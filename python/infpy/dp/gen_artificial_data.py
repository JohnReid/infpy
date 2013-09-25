#
# Copyright John Reid 2010
#


"""
Create artificial test data from a HDPM.
"""

from infinite_multinomials import LazyInfiniteSequence, StickBreaker, InfiniteDirichlet, plot_multinomials
import numpy as N, numpy.random as R, itertools, cookbook.pylab_utils as pylab_utils
import pylab as P, logging


def seed(s):
    R.seed(s)
    from rpy2.robjects import r
    x = r['set.seed'](s)


def rGamma(alpha, beta):
    "Sample from gamma distribution."
    from rpy2.robjects import r
    x = r.rgamma(1, alpha, beta)
    if N.isnan(x):
        raise RuntimeError('Cannot draw from gamma distribution with parameters: (%f, %f)' % (alpha, beta))
    return x[0]


def rDirichlet(alpha):
    "Sample from Dirichlet distribution."
    x = R.mtrand.dirichlet(alpha)
    if N.isnan(x).any():
        raise RuntimeError('Cannot draw from Dirichlet distribution')
    return x


def rMultinomial(alpha):
    "Sample from multinomial distribution."
    x = R.multinomial(1, alpha)
    if N.isnan(x).any():
        raise RuntimeError('Cannot draw from multinomial distribution')
    return N.where(x)[0][0]


class HDPMSampler(object):
    """
    Samples from a HDPM.
    """

    def __init__(
        self,
        document_sizes,
        a_alpha, b_alpha,
        a_beta,  b_beta,
        a_gamma, b_gamma,
        a_tau
    ):
        "Initialise with hyperparameters."
        self.document_sizes = document_sizes
        self.a_alpha = a_alpha
        self.b_alpha = b_alpha
        self.a_beta = a_beta
        self.b_beta = b_beta
        self.a_gamma = a_gamma
        self.b_gamma = b_gamma
        self.a_tau = a_tau
        self.sample_parameters()
        self.sample_programs()
        self.sample_data()

    def D(self):
        "@return: The number of documents in the sample."
        return len(self.document_sizes)

    def K(self):
        "@return: The number of programs used in the sample."
        return max(
            max(Z)
            for theta, Z, X in self.documents
        ) + 1

    def N(self):
        "@return: The number of words in the sample."
        return sum(self.document_sizes)

    def W(self):
        "@return: The number of words in the vocabulary."
        return len(self.a_tau)

    def sample_parameters(self):
        "Sample parameters."
        self.alpha = rGamma(self.a_alpha, self.b_alpha)
        self.beta  = rGamma(self.a_beta,  self.b_beta )
        self.gamma = rGamma(self.a_gamma, self.b_gamma)
        self.tau = rDirichlet(self.a_tau)
        self.pi = StickBreaker(self.gamma)

    def sample_programs(self):
        "Sample programs."
        self.programs = LazyInfiniteSequence(extender=self.sample_program)

    def sample_program(self):
        "Sample one program."
        return rDirichlet(self.beta * self.tau)

    def sample_document(self, d):
        "Sample a document."
        length = self.document_sizes[d]
        theta = InfiniteDirichlet(self.alpha, self.pi)
        Z = [theta.draw() for i in xrange(length)]
        X = [rMultinomial(self.programs[z]) for z in Z]
        return theta, Z, X

    def sample_data(self):
        "Sample data points."
        self.documents = [self.sample_document(d) for d in xrange(self.D())]
    
    def get_documents(self):
        return [X for theta, Z, X in self.documents]



def summarise_sample(sampler):
    "Produce various plots to summarise the sample."

    logging.info(
        'Sampled from HDPM: W=%d; D=%d; N=%d; K=%d',
        sampler.W(), sampler.D(), sampler.N(), sampler.K()
    )

    #
    # Plot Pi and the thetas
    #
    P.figure()
    plot_multinomials(
        list(itertools.chain((sampler.pi,), [t for t, Z, X in sampler.documents])),
        itertools.imap(
            lambda c, i: {
                'color':c,
                'label':i and 'theta[%d]' % (i-1) or 'pi'
            },
            itertools.cycle(pylab_utils.simple_colours),
            itertools.count()
        )
    )
    P.legend(loc='top right')
    P.title('Pi and thetas')

    #
    # Plot the programs
    #
    P.figure()
    spacing = .3
    K = sampler.K()
    W = sampler.W()
    width = (1. - spacing) / K
    for k, program in enumerate(sampler.programs):
        P.bar(N.arange(W) + (-width*K/2 + k*width), program, width)
    P.legend()
    P.title('Programs')



if '__main__' == __name__:

    logging.basicConfig(level=logging.INFO)

    seed(2)
    W = 6
    sampler = HDPMSampler(
        N.arange(5)+10,
        10., 20.,
        20., 10.,
        20., 10.,
        N.ones(W)
    )

    summarise_sample(sampler)
    P.show()
