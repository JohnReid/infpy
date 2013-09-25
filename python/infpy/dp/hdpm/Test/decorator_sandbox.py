#
# Copyright John Reid 2010
#


"""
Code to test a decorator that caches a return value where the cache can be cleared.
"""

from cookbook.decorate import simple_decorator
import logging

logging.basicConfig(level=logging.INFO)


class Memoize(object):

    def __init__(self,function):
        self._callable = function

    def __call__(self):
        try:
            return self.cached_value
        except AttributeError:
            self.cached_value = self._callable()
            return self.cached_value

memoize = simple_decorator(Memoize)



class Klass(object):

    def __init__(self):
        self._calculate_value = memoize(self._calculate_value)

    def _calculate_value(self):
        logging.info('id=%d: Calculating value', id(self))
        return 5 * 3

    def _get_value(self):
        return self._calculate_value()

    value = property(_get_value, "I'm the class's value.")

    def log_value(self):
        logging.info(self.value)




o = Klass()
o.log_value()
o.log_value()
del o._calculate_value.cached_value
o.log_value()
o.log_value()

logging.info('')

o2 = Klass()
o2.log_value()
o2.log_value()
del o2._calculate_value.cached_value
o.log_value()
o.log_value()
o2.log_value()
o2.log_value()
