#
# Copyright John Reid 2006
#

import infpy
import numpy

def kernel_init_args_helper(
        params = None,
        priors = None,
        dimensions = None,
        num_extra_params = 0
):
    """
    A helper function to allow user to only specify some of the priors,
    parameters and dimensions arguments.

    :param params: the parameters
    :param priors: the priors over the parameters
    :param dimensions: the number of dimensions of the input values
    :param num_extra_params: the number of extra parameters (over and above one per
        dimension)

    This function fills any unspecified arguments in with defaults.

    It returns (params, priors, dimensions)
    """
    # Try and work out dimensions if not specified
    if None == dimensions:
        if None == params:
            if None == priors: dimensions = 1 # assume dims = 1
            else: dimensions = len(priors) - num_extra_params
        else:
            dimensions = len(params) - num_extra_params
    if 1 > dimensions: raise RuntimeError('# dimensions must be positive')

    # Use defaults if params and/or priors not set
    if None == params:
        params = numpy.ones( dimensions + num_extra_params, numpy.float64 )
    if None == priors:
        priors = [ None ] * (dimensions + num_extra_params)
    params = numpy.asarray( params, numpy.float64 )

    # Check params and priors agree in length and with # dimensions
    if len(params) != len(priors):
        raise RuntimeError( """Must have same number of priors (%d) """
                """as parameters (%d))""" % ( len(priors), len(params) ) )
    if len(params) != dimensions + num_extra_params:
        raise RuntimeError( """Must supply %d more parameter(s) (%d) than """
                """dimensions (%d).""" % (
                num_extra_params, len(params), dimensions
        ) )

    # all done
    return ( params, priors, dimensions )



class Kernel( object ):
    """
    Base class for all Gaussian process kernels
    """

    def __init__(
            self,
            params,
            param_priors
    ):
        if len(params) != len(param_priors):
            raise RuntimeError( """Must have same number of priors as parameters""" )
        self.params = params
        self.param_priors = param_priors
        try:
            # if it has a derivative class, create derivative object for each parameter
            self.__class__.Derivative
            self.derivatives = [
                self.__class__.Derivative( self, i )
                for i in xrange( len(params) )
            ]
        except:
            pass


    def __mul__(self, k_rhs):
        """
        @return: The product of self and k_rhs
        """
        return ProductKernel(self, k_rhs)


    def __add__(self, k_rhs):
        """
        @return: The sum of self and k_rhs
        """
        return SumKernel(self, k_rhs)


    def set_params( self, new_params ):
        """
        Set the parameters of the kernel
        """
        if len( new_params ) != len( self.params ):
            raise RuntimeError( """Wrong number of parameters supplied (%d). """
            """Was expecting %d"""
                    % ( len( new_params ), len( self.params ) ) )
        for i, p in enumerate( new_params ):
            self.params[i] = p


    def supports( self, p ):
        """
        True if all the parameters, p, are in the support of the kernel's priors
        """
        if len( p ) != len( self.param_priors ):
            raise RuntimeError( """Must supply same number of parameters as the"""
                    """kernel has priors""" )
        for p1, prior in zip( p, self.param_priors ):
            if None != prior and not prior.supports( p1 ): return False
        return True


    def __str__( self ):
        """String representation of kernel"""
        raise RuntimeError( """%s.__str__() Should have been implemented """
                """in base class""" % str(self.__class__) )


    def __call__( self, x1, x2, identical = False ):
        """Returns value of kernel for the specified data points

        If identical == True x1 and x2 are the same sample
        """
        raise RuntimeError, 'Kernel.__call__() should be implemented in sub-class'


    def derivative_wrt_param( self, i ):
        """
        Returns the derivative of the kernel with respect to the i'th parameter.
        """
        return self.derivatives[i]



class ConstantKernel( Kernel ):
    """
    A constant kernel.
    """

    def __init__( self, constant = 1.0, prior = infpy.GammaDistribution() ):
        """Initialise the constant kernel
        """
        import types
        Kernel.__init__( self, numpy.asarray([constant], numpy.float64), [ prior ] )

    def __str__( self ):
        return "ConstantKernel( %f )" % self.params[0]

    def __call__( self, x1, x2, identical = False ):
        return self.params[0]

    class Derivative( object ):
        def __init__( self, k, i ):
            self.k = k
            assert 0 == i
        def __call__( self, x1, x2, identical = False ):
            return 1.0


class ConstantSquaredKernel( Kernel ):
    def __init__( self, constant = 1.0, prior = infpy.LogNormalDistribution() ):
        """
        Initialise the constant kernel

        @arg constant: Actually the square root of the constant
        @arg prior: Actually the prior of the square root of the constant
        """
        import types
        Kernel.__init__( self, numpy.asarray([constant], numpy.float64), [ prior ] )

    def __str__( self ):
        return "ConstantSquaredKernel( %f )" % self.params[0] ** 2

    def __call__( self, x1, x2, identical = False ):
        return self.params[0] ** 2

    class Derivative( object ):
        def __init__( self, k, i ):
            self.k = k
            assert 0 == i
        def __call__( self, x1, x2, identical = False ):
            return 2 * self.k.params[0]


class IdenticalKernel( Kernel ):
    """
    1 if x1 and x2 are identical, 0 otherwise
    """

    def __init__( self ):
        Kernel.__init__( self, numpy.asarray([ ]), [ ] )

    def __str__( self ):
        return "IdenticalKernel"

    def __call__( self, x1, x2, identical = False ):
        if identical: return 1.0
        else: return 0.0



class EqualityKernel( Kernel ):
    """
    1 if x1 and x2 are equal, 0 otherwise
    """

    def __init__( self ):
        Kernel.__init__( self, numpy.asarray([ ]), [ ] )

    def __str__( self ):
        return "EqualityKernel"

    def __call__( self, x1, x2, identical = False ):
        if x1 == x2: return 1.0
        else: return 0.0


class KernelParameterFixer( Kernel ):
    """
    A wrapper kernel that hides (i.e. fixes) the parameters of another kernel

    To users of this class it appears as the kernel has no parameters to optimise.
    This can be useful when you have a mixture kernel and you only want to learn
    one child kernel's parameters.
    """

    def __init__( self, hidden_k ):
        Kernel.__init__( self, numpy.asarray([ ]), [ ] )
        self.hidden_k = hidden_k

    def __str__( self ):
        return "ParameterFixerKernel"

    def __call__( self, x1, x2, identical = False ):
        return self.hidden_k( x1, x2, identical )



def noise_kernel( sigma = 1.0, sigma_prior = infpy.LogNormalDistribution() ):
    return IdenticalKernel() * ConstantSquaredKernel( sigma, sigma_prior )


from sum_kernel import *
