#
# Copyright John Reid 2006
#


from kernel import *
import infpy
import numpy
import math


class IndexMixer(object):
    """Mixes two indexable objects to provide one seamlessly indexable object

    That is len(IndexMixer(x1,x2))=len(x1)+len(x2)
    """

    def __init__(self, o1, o2):
        self.o1 = o1
        self.o2 = o2

    def _check_key(self, key):
        if int != type(key):
            raise TypeError('Need integral key')
        if 0 > key >= len(self):
            raise IndexError('Index out of range')

    def _get_container_and_index(self, key):
        self._check_key(key)
        if key < len(self.o1):
            return (self.o1, key)
        else:
            return (self.o2, key - len(self.o1))

    def __len__(self):
        return len(self.o1) + len(self.o2)

    def __getitem__(self, key):
        o, k = self._get_container_and_index(key)
        return o.__getitem__(k)

    def __setitem__(self, key, value):
        o, k = self._get_container_and_index(key)
        return o.__setitem__(k, value)

    def __delitem__(self, key):
        o, k = self._get_container_and_index(key)
        return o.__delitem__(k)

    class Iter(object):
        def __init__(self, mixer):
            self.mixer = mixer
            self.i = 0

        def next(self):
            self.i += 1
            if self.i > len(self.mixer):
                raise StopIteration
            return self.mixer[self.i - 1]

    def __iter__(self):
        return IndexMixer.Iter(self)
        # raise RuntimeError, '__iter__() not implemented'

    def __str__(self):
        return '[ %s ]' % ", ".join(str(i) for i in self)


class MixKernel(Kernel):
    """Kernel that mixes two other kernels. E.g. used as sub-class for SumKernel
    """

    def __init__(self, k1, k2):
        Kernel.__init__(
            self,
            IndexMixer(k1.params, k2.params),
            IndexMixer(k1.param_priors, k2.param_priors)
        )
        self.k1 = k1
        self.k2 = k2

    def _get_kernel_and_index(self, key):
        """Returns a tuple:

        (kernel the key refers to, other kernel, index into this kernel)
        """
        if key < len(self.k1.params):
            return (self.k1, self.k2, key)
        else:
            return (self.k2, self.k1, key - len(self.k1.params))


class SumKernel(MixKernel):
    """The sum of 2 other kernels"""

    def __init__(self, k1, k2):
        MixKernel.__init__(self, k1, k2)

    def __call__(self, x1, x2, identical=False):
        return self.k1(x1, x2, identical) + self.k2(x1, x2, identical)

    def __str__(self):
        return """SumKernel( %s + %s )""" % (str(self.k1), str(self.k2))

    def derivative_wrt_param(self, index):
        k, other_k, i = self._get_kernel_and_index(index)
        return k.derivative_wrt_param(i)


class ProductKernel(MixKernel):
    """The product of 2 other kernels"""

    def __init__(self, k1, k2):
        MixKernel.__init__(self, k1, k2)

    def __call__(self, x1, x2, identical=False):
        # print x1,x2,identical,self.k1( x1, x2, identical ),self.k2( x1, x2, identical )
        return (
            self.k1(x1, x2, identical)
            * self.k2(x1, x2, identical)
        )

    def __str__(self):
        return """ProductKernel( %s * %s )""" % (str(self.k1), str(self.k2))

    def derivative_wrt_param(self, index):
        k, other_k, i = self._get_kernel_and_index(index)
        dp = k.derivative_wrt_param(i)
        return lambda x1, x2, identical = False: other_k(x1, x2, identical) * dp(x1, x2, identical)
