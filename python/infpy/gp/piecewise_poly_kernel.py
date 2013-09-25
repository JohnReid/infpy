#
# Copyright John Reid 2006, 2012
#


from real_kernel import *
import numpy, math

class PiecewisePolyKernel( RealKernel ):
    """Eq 4.21 in `Gaussian Processes for Machine Learning`__ by Rasmussen and Williams. 

    __ http://www.amazon.co.uk/Gaussian-Processes-Learning-Adaptive-Computation/dp/026218253X/

    Base class for specialisations for specific values of q

    Will produce a sparse covariance matrix with 0's where r > 1
    """

    def __init__(
            self,
            q,
            params = None,
            priors = None,
            dimensions = None
    ):
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
        self.q = int(q)
        self.j = int(dimensions) / 2 + self.q + 1


class PiecewisePoly0Kernel( PiecewisePolyKernel ):
    """Eq 4.21 in `Gaussian Processes for Machine Learning`__ by Rasmussen and Williams. 

    __ http://www.amazon.co.uk/Gaussian-Processes-Learning-Adaptive-Computation/dp/026218253X/
    
    q=0
    """

    def __init__(
            self,
            params = None,
            priors = None,
            dimensions = None
    ):
        PiecewisePolyKernel.__init__( self, 0, params, priors, dimensions )

    def __str__( self ):
        return """PiecewisePolyKernel( q = 0 )"""

    def __call__( self, x1, x2, identical = False ):
        (x1, x2) = self._check_args( x1, x2 )
        r = distance( x1, x2, self.params )
        if r >= 1.0: return 0
        return (
                ( 1.0 - r ) ** self.j
        )

    class Derivative( object ):
        def __init__( self, k, i ):
            self.k = k
            self.i = i
        def __call__( self, x1, x2, identical = False ):
            (x1, x2) = self.k._check_args( x1, x2 )
            r = distance( x1, x2, self.k.params )
            if r >= 1.0: return 0
            return (
                    - self.k.j * (1.0 - r) ** (self.k.j - 1)
            ) * distance_derivative( x1, x2, self.k.params, self.i )

class PiecewisePoly1Kernel( PiecewisePolyKernel ):
    """Eq 4.21 in `Gaussian Processes for Machine Learning`__ by Rasmussen and Williams. 

    __ http://www.amazon.co.uk/Gaussian-Processes-Learning-Adaptive-Computation/dp/026218253X/
    
    q=1
    """

    def __init__(
            self,
            params = None,
            priors = None,
            dimensions = None
    ):
        PiecewisePolyKernel.__init__( self, 1, params, priors, dimensions )

    def __str__( self ):
        return """PiecewisePolyKernel( q = 1 )"""

    def __call__( self, x1, x2, identical = False ):
        (x1, x2) = self._check_args( x1, x2 )
        r = distance( x1, x2, self.params )
        if r >= 1.0: return 0
        return (
                ( 1.0 - r ) ** (self.j + 1)
        ) * ((self.j + 1) * r + 1.0 )

    class Derivative( object ):
        def __init__( self, k, i ):
            self.k = k
            self.i = i
        def __call__( self, x1, x2, identical = False ):
            (x1, x2) = self.k._check_args( x1, x2 )
            r = distance( x1, x2, self.k.params )
            if r >= 1.0: return 0
            return (
                    (1.0 - r) ** self.k.j
                    * (
                            (1 + self.k.j) * (1.0 - r)
                            + ((-self.k.j**2  - 2 * self.k.j - 1) * r - self.k.j - 1.0)
                    )
            ) * distance_derivative( x1, x2, self.k.params, self.i )
