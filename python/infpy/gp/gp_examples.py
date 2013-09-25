#
# Copyright John Reid 2009
#

import numpy, math, pylab, infpy, sys, scipy.stats

def gp_ex_fixed_period_1():
    """Example of the fixed period kernel"""
    start, end = 0.0, 4.0
    X = infpy.gp_1D_X_range( start, end, 0.4 )
    y = numpy.asarray( [ math.sin( 2.0 * math.pi * x[0] ) for x in X ] )
    # pylab.plot( [ x[0] for x in X ], [ y1 for y1 in y ] )
    # pylab.show()
    # return
    LN = infpy.LogNormalDistribution
    Gamma = infpy.GammaDistribution
    k = (
            infpy.noise_kernel( 0.1, LN(  ) )
            + infpy.ConstantKernel( 1.0, Gamma(  ) )
            * infpy.FixedPeriod1DKernel( 1.0 )
    )
    gp = infpy.GaussianProcess( X, y, k )
    infpy.gp_learn_hyperparameters( gp )
    infpy.gp_1D_predict( gp )
# gp_ex_fixed_period_1()

def gp_ex_fixed_period():
    """Example of the fixed period kernel"""
    start, end = -4.0, 0.0
    X = infpy.gp_1D_X_range( start, end, 1.3 )
    # X = [ ]
    # X = [ [ 0.0 ] ]
    # X = [ [ 0.0 ], [ 1.0 ] ]
    # X = [ [ 0.0 ], [ -1.0 ], [ -2.0 ], [ -3.0 ], ]
    y = numpy.asarray( [ math.sin( 2.0 * math.pi * x[0] ) for x in X ] )
    # pylab.plot( [ x[0] for x in X ], [ y1 for y1 in y ] )
    # pylab.show()
    LN = infpy.LogNormalDistribution
    k = (
            infpy.FixedPeriod1DKernel( 1.0 )
            + infpy.noise_kernel( 0.1 )
    )
    gp = infpy.GaussianProcess( X, y, k )
    sample_X = infpy.gp_1D_X_range( start, end, 0.03 )
    y = infpy.gp_sample_from( gp, sample_X )
    ( y, V_f_star, log_p_y_given_X ) = gp.predict( sample_X )
    infpy.gp_plot_prediction( sample_X, y )
    infpy.gp_title_and_show( gp )
# gp_ex_fixed_period()

def gp_ex_output_scale():
    """Examine how kernel parameters should change for scaled outputs"""
    start, end = 0.0, 1.0
    X = infpy.gp_1D_X_range( start, end, 0.04 )

    noise_level = 0.1
    small_y = infpy.gp_zero_mean(
            numpy.asarray(
                    [ x[0]**2 + noise_level * scipy.stats.norm().rvs()[0] for x in X ]
            )
    )
    # big_y = small_y * 100.0

    LN = infpy.LogNormalDistribution
    k = (
            infpy.noise_kernel( noise_level, LN( math.log( math.sqrt( noise_level ) ) ) )
            + infpy.ConstantSquaredKernel( 1.0, LN( math.log( 1.0 ) ) )
            * infpy.SquaredExponentialKernel( [ 1.0 ], [ LN() ] )
    )
    gp = infpy.GaussianProcess( X, small_y, k )

    infpy.gp_1D_predict( gp, 100 )

    infpy.gp_learn_hyperparameters( gp )
    infpy.gp_1D_predict( gp, 100 )
# gp_ex_output_scale()

def gp_ex_fig_22():
    """Similar to figure 2.2 in `Gaussian Processes for Machine Learning`__ by Rasmussen and Williams. 

    __ http://www.amazon.co.uk/Gaussian-Processes-Learning-Adaptive-Computation/dp/026218253X/
    """
    X = [
            [ -4.0 ],
            [ -3.0 ],
            [ -1.0 ],
            [  0.0 ],
            [  2.0 ],
    ]
    y = numpy.array( [
            -2.0,
            0.0,
            1.0,
            2.0,
            -1.0
    ] )
    LN = infpy.LogNormalDistribution
    Gamma = infpy.GammaDistribution
    k = (
            infpy.noise_kernel( .2, Gamma( 1.0, 1.0 ) )
            + infpy.ConstantKernel( )
            * infpy.SquaredExponentialKernel( [ 1.0 ], [ LN() ] )
    )
    gp = infpy.GaussianProcess( X, y, k )
    #infpy.gp_1D_predict( gp, 100, x_min = -5.0, x_max = 5.0 )
    infpy.gp_learn_hyperparameters( gp )
    infpy.gp_1D_predict( gp, 100, x_min = -5.0, x_max = 5.0 )
# gp_ex_fig_22()
