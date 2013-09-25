#
# Copyright John Reid 2006, 2012
#


from kernel import *
from real_kernel import *
import numpy, math


class Matern32Kernel( RealKernel ):
    """Eq 4.17 in `Gaussian Processes for Machine Learning`__ by Rasmussen and Williams. 

    __ http://www.amazon.co.uk/Gaussian-Processes-Learning-Adaptive-Computation/dp/026218253X/
    """

    _root_three = math.sqrt( 3.0 )

    def __init__( self, params = None, priors = None, dimensions = None ):
        ( params, priors, dimensions ) = kernel_init_args_helper(
                params,
                priors,
                dimensions
        )
        RealKernel.__init__(
                self,
                params,
                priors,
                dimensions
        )

    def __str__( self ):
        return """MaternKernel32"""

    def __call__( self, x1, x2, identical = False ):
        (x1, x2) = self._check_args( x1, x2 )
        r = Matern32Kernel._root_three * distance( x1, x2, self.params )
        return ( 1.0 + r ) * math.exp( - r )

    class Derivative( object ):
        def __init__( self, k, i ):
            self.k = k
            self.i = i
        def __call__( self, x1, x2, identical = False ):
            (x1, x2) = self.k._check_args( x1, x2 )
            r = Matern32Kernel._root_three * distance( x1, x2, self.k.params )
            dk_dr = -r * math.exp( -r )
            dr_dparam = (
                    Matern32Kernel._root_three
                    * distance_derivative( x1, x2, self.k.params, self.i )
            )
            return dk_dr * dr_dparam




class Matern52Kernel( RealKernel ):
    """Eq 4.17 in `Gaussian Processes for Machine Learning`__ by Rasmussen and Williams. 

    __ http://www.amazon.co.uk/Gaussian-Processes-Learning-Adaptive-Computation/dp/026218253X/
    """

    _root_five = math.sqrt( 5.0 )

    def __init__( self, params = None, priors = None, dimensions = None ):
        ( params, priors, dimensions ) = kernel_init_args_helper(
                params,
                priors,
                dimensions
        )
        RealKernel.__init__(
                self,
                params,
                priors,
                dimensions
        )

    def __str__( self ):
        return """MaternKernel52"""

    def __call__( self, x1, x2, identical = False ):
        (x1, x2) = self._check_args( x1, x2 )
        r = distance( x1, x2, self.params )
        r_root_5 = Matern52Kernel._root_five * r
        return (
                3.0 * r_root_5 + r_root_5 ** 2 + 3.0
        ) * math.exp( - r_root_5 ) / 3.0

    class Derivative( object ):
        def __init__( self, k, i ):
            self.k = k
            self.i = i
        def __call__( self, x1, x2, identical = False ):
            (x1, x2) = self.k._check_args( x1, x2 )
            r = distance( x1, x2, self.k.params )
            return (
                    - Matern52Kernel._root_five * 5.0 * r ** 2
                    - 5.0 * r
            ) * math.exp( - Matern52Kernel._root_five * r ) / 3.0 \
            * distance_derivative( x1, x2, self.k.params, self.i )
