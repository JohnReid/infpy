
import rpy
import infpy
import numpy

def r_chi_square_example():
    degrees = 4
    grid = rpy.r.seq(0, 10, length=100)
    values = [rpy.r.dchisq(x, degrees) for x in grid]
    rpy.r.par(ann=0)
    rpy.r.plot(grid, values, type='lines')


def gp_on_nhtemp_ex():
    y = numpy.array( rpy.r.nhtemp )
    y -= numpy.mean( y )
    X = [ [ i ] for i in range( len( y ) ) ]
    # import pylab
    # pylab.plot( y )
    # pylab.show()
    # print X
    # print y
    K = (
            infpy.SquaredExponentialKernel( dimensions = 1 )
            + infpy.noise_kernel( 0.4 )
    )
    gp = infpy.GaussianProcess( X, y, K )
    infpy.gp_1D_predict( gp, num_steps = 120 )
    # infpy.learn_gp_hyperparameters( gp )
    # infpy.gp_1D_predict( gp, num_steps = 120 )
gp_on_nhtemp_ex()
