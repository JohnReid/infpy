#
# Copyright John Reid 2006
#


from kernel import *
import numpy


class PreCalculatedKernel(Kernel):
    """A kernel that has been pre-computed"""

    def __init__(
            self,
            objects,
            matrix
    ):
        """objects: The objects that we have pre-computed the kernel on
        matrix: The kernel values
        """
        self.objects = objects
        self.matrix = numpy.array(matrix, copy=True)
        if len(self.matrix.shape) != 2:
            raise RuntimeError('Kernel must be of rank 2')
        if self.matrix.shape[0] != self.matrix.shape[1]:
            raise RuntimeError('Kernel must be square')
        if self.matrix.shape[0] != len(objects):
            raise RuntimeError('Kernel must be of same size as objects')
        self.indices = dict(
            [
                (o, i)
                for i, o in enumerate(objects)
            ]
        )
        Kernel.__init__(self, [], [])

    def __str__(self):
        return """PreCalculatedKernel"""

    def __call__(self, x1, x2, identical=False):
        if identical and x1 != x2:
            raise RuntimeError('x1 and x2 should be identical')
        return self.matrix[
            self.indices[x1],
            self.indices[x2]
        ]
