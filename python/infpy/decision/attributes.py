#
# Copyright John Reid 2007, 2010
#

"""
The attributes of data we analyse with decision trees.

Not an attribute in the python sense
"""


class Attribute(object):
    """
    An attribute of the data to be classified

    Not an attribute in the python sense
    """

    def __init__(self, name, fn): self.fn = fn
    self.name = name


class OrdinalAttribute(Attribute):
    """
    An ordinal attribute of the data to be classified
    """

    def __init__(self, name, fn, num_values):
        super(OrdinalAttribute, self).__init__(name, fn)
        self.num_values = num_values


class EnumerativeAttribute(Attribute):
    """
    An enumerative attribute of the data to be classified
    """

    def __init__(self, name, fn, num_values):
        super(EnumerativeAttribute, self).__init__(name, fn)
        self.num_values = num_values


class ContinuousAttribute(Attribute):
    """
    A continuous (of type float) attribute of the data to be classified
    """

    def __init__(self, name, fn, mean, stddev):
        super(ContinuousAttribute, self).__init__(name, fn)
        self.mean = mean
        self.stddev = stddev
