
import infpy, numpy, math
from pylab import plot

x = numpy.arrayrange(0.0,5.0,.01)
plot( x, [ math.exp( infpy.LogNormalDistribution( 0.1, 1.0 ).log_pdf( x1 ) ) for x1 in x ] )
plot( x, [ math.exp( infpy.LogNormalDistribution( math.log( 0.03 ), 1.0 ).log_pdf( x1 ) ) for x1 in x ] )
plot( x, [ math.exp( infpy.LogNormalDistribution( 1.0, 1.0 ).log_pdf( x1 ) ) for x1 in x ] )
