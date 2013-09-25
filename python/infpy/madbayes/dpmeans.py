#
# Copyright John Reid 2013
#


"""
The DP-means algorithm by `Kulis et al.`_

.. _Kulis et al.: http://arxiv.org/abs/1111.0352    
"""

import logging, numpy
from itertools import count
from numpy import newaxis


def plot_clusters(x, z, format_cycler=None):
    import pylab as P
    from cookbook.pylab_utils import create_format_cycler, simple_marker_styles, simple_colours

    if not format_cycler:
        format_cycler = create_format_cycler(marker=simple_marker_styles, color=simple_colours)
        
    clusters = set(z)
    for i, c in enumerate(clusters):
        cluster_x = x[z==c]
        P.scatter(cluster_x[:,0], cluster_x[:,1], **format_cycler(i))
    

def dpmeans(x, lambda_, progress_plots=False):
    """
    The DP-means algorithm by `Kulis et al.`_
    
    .. _Kulis et al.: http://arxiv.org/abs/1111.0352    
    
    
    
    :parameters:
    
    - :math:`x` : input data, a sequence of length :math:`N`
    - :math:`\lambda` : cluster penalty parameter
    - progress_plots : Scatter plot clusters at every iteration
    
    :returns: Cluster indicator variables
    

    Algorithm:
    
    #. Initialise:
      
      * Number of clusters :math:`K=1`
      * Global cluster mean :math:`\\mu_1 = \\frac{1}{n} \\sum_n x_n`
      * Cluster indicator variables :math:`z_n = 0 \\quad \\forall n`
      
    #. Repeat until convergence:
    
      * For each point :math:`n`:

        - Compute distance to each cluster :math:`d_{nk} = ||x_n - \\mu_k||^2`
        - If :math:`\\min d_{nk} > \lambda` then set :math:`K=K+1, z_n=K, \\mu_k=x_n`
        - Otherwise set :math:`z_n= \\arg\!\\min_k d_{nk}`

      * For each cluster :math:`k`, compute :math:`\\mu_k = \\frac{1}{|\{n: z_n = k\}|}\sum_{n: z_n = k} x_n`
    
    """
    if progress_plots:
        import pylab as P
        from cookbook.pylab_utils import pylab_context_ioff, \
            create_format_cycler, simple_marker_styles, simple_colours
        format_cycler = create_format_cycler(marker=simple_marker_styles, color=simple_colours)
    
    N = len(x)
    logging.info('Got %d data', N)
    lambda2 = lambda_ ** 2
    z = numpy.zeros(N, dtype=numpy.int) # initialise cluster indicators
    last_z = None
    for i in count(1):
    #for i in xrange(1,21):
        logging.info('Iteration %d: have %d cluster(s)', i, int(z.max() + 1))
        
        # calculate cluster means
        mu = [numpy.mean(x[z==k], axis=0) for k in xrange(int(z.max() + 1))]

        for n, xn in enumerate(x):
            d2 = numpy.array([((xn - muk)**2).sum() for muk in mu])
            closest_k = d2.argmin()
            if d2[closest_k] > lambda2:
                mu.append(xn)
            else:
                z[n] = closest_k
        
        # make clusters contiguous from 0
        cluster_map = dict((c, k) for k, c in enumerate(set(z)))
        if len(cluster_map) < int(z.max() + 1):
            logging.warning('Reducing cluster indices')
        for n, c in enumerate(z):
            z[n] = cluster_map[c]
        
        # check if we have converged by testing if no z have changed
        if None != last_z and (z == last_z).all():
            break
        last_z = z.copy()
        
        if progress_plots:
            with pylab_context_ioff():
                P.figure()
                plot_clusters(x, z, format_cycler)
                P.savefig('dpmeans-%04d.png' % i)
                P.close()
    
    num_clusters = len(set(z))
    logging.info('Have %d cluster(s)', num_clusters)
    return z
