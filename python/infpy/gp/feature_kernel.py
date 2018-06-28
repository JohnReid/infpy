#
# Copyright John Reid 2006
#

from kernel import *
import numpy


class AttributeExtractor(Kernel):
    """
    Changes the type of object a kernel can act on

    For example our data might be of type X that aggregate several features:

    X.vec: a numpy.array
    X.is_true: a boolean

    We may have a kernel, Kvec, that acts on arrays and a kernel, Kbool,
    that acts on boolean data. We could combine them as follows::

        K = SumKernel(
            AttributeExtractor( 'vec', Kvec ),
            AttributeExtractor( 'is_true', Kbool )
        )

    The AttributeExtractor will find the vec and is_true attributes of
    each x and pass them to the kernels Kvec and Kbool
    """

    def __init__(self, attribute_name, sub_kernel):
        """Creates a kernel that extracts attributes with the given name and pass
        them to the sub kernel
        """
        Kernel.__init__(self, sub_kernel.params, sub_kernel.param_priors)
        self.k = sub_kernel
        self.attribute_name = attribute_name

    def __str__(self):
        return """AttributeExtractorKernel( %s )""" % self.attribute_name

    def __call__(self, x1, x2, identical=False):
        return self.k(
            getattr(x1, self.attribute_name),
            getattr(x2, self.attribute_name),
            identical
        )

    class Derivative(object):
        def __init__(self, k, i):
            self.k = k
            self.i = i

        def __call__(self, x1, x2, identical=False):
            return self.k.k.derivative_wrt_param(self.i)(
                x1.__dict__[self.k.attribute_name],
                x2.__dict__[self.k.attribute_name],
                identical
            )
