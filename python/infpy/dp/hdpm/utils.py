#
# Copyright John Reid 2008, 2009, 2010
#


"""
Utility functions for HDPM code.
"""


import logging
import cookbook
from optparse import OptionGroup
from cookbook.decorate import simple_decorator


def add_hdpm_options(parser):
    "Add options for the HDPM to option parser."

    hyper_group = OptionGroup(
        parser,
        "Hyper-parameter options",
        "Options to set the hyper-parameters."
    )
    hyper_group.add_option(
        "--a-gamma",
        dest="a_gamma",
        default=7.5,
        type='float',
        help="Hyper-parameter for gamma which controls number of topics."
    )
    hyper_group.add_option(
        "--b-gamma",
        dest="b_gamma",
        default=7.5,
        type='float',
        help="Hyper-parameter for gamma which controls number of topics."
    )
    hyper_group.add_option(
        "--a-beta",
        dest="a_beta",
        default=7.5,
        type='float',
        help="Hyper-parameter for beta which controls diversity of topics."
    )
    hyper_group.add_option(
        "--b-beta",
        dest="b_beta",
        default=7.5,
        type='float',
        help="Hyper-parameter for beta which controls diversity of topics."
    )
    hyper_group.add_option(
        "--a-alpha",
        dest="a_alpha",
        default=7.5,
        type='float',
        help="Hyper-parameter for alpha which controls diversity of document-specific topic distributions."
    )
    hyper_group.add_option(
        "--b-alpha",
        dest="b_alpha",
        default=7.5,
        type='float',
        help="Hyper-parameter for alpha which controls diversity of document-specific topic distributions."
    )
    parser.add_option_group(hyper_group)


def log_stack(level=logging.INFO, limit=None, tag=None):
    import traceback
    if None != limit:
        limit += 1
    for line in traceback.format_list(traceback.extract_stack(limit=limit))[:-1]:
        line = line.strip()
        logging.log(level, None != tag and '%s: %s' % (tag, line) or line)


def log_stack_simple(level=logging.INFO, limit=None, tag=None):
    import traceback
    if None != limit:
        limit += 1
    logging.log(level, 'STARTING STACK')
    for filename, line_number, function_name, text in traceback.extract_stack(limit=limit)[:-1]:
        logging.log(level, 'Stack: %s', None != tag and '%s:\t%s' %
                    (tag, function_name) or '\t%s' % function_name)
    logging.log(level, 'ENDING STACK')


def check_stack_for_multiple_calculates():
    "Examines stacks to see if we have called a _calculate_ function recursively. For debugging purposes."
    import traceback
    stack = list(filter(lambda e: e[2].startswith(
        '_calculate_'), traceback.extract_stack()))
    if len(stack) > 1:
        log_stack()
        1 / 0


class ProtectAgainstRecursion(object):
    "A decorator to protect against recursion."

    def __init__(self, fn):
        self._callable = fn
        self.in_call = False

    def __call__(self, *args, **kwds):
        if self.in_call:
            raise RuntimeError('Identified recursion!')
        self.in_call = True
        try:
            result = self._callable(*args, **kwds)
        finally:
            self.in_call = False
        return result


protect_against_recursion = simple_decorator(ProtectAgainstRecursion)


class Memoize(object):

    def __init__(self, function):
        self._callable = function
        self.set_cache_enabled(True)
        self.logging_level = None

    def set_cache_enabled(self, value):
        self._cache_enabled = value

    def has_cached_value(self):
        try:
            self.cached_value
            return True
        except AttributeError:
            return False

    def __call__(self):
        if not self._cache_enabled:
            if None != self.logging_level:
                logging.log(
                    self.logging_level, 'Memoize: %s: cache disabled.', self._callable.__name__)
            return self._callable()
        else:
            try:
                value = self.cached_value
                if None != self.logging_level:
                    logging.log(
                        self.logging_level, 'Memoize: %s: using cached value.', self._callable.__name__)
                return value
            except AttributeError:
                if None != self.logging_level:
                    logging.log(
                        self.logging_level, 'Memoize: %s: calculating cached value.', self._callable.__name__)
                self.cached_value = self._callable()
                return self.cached_value

    def clear_cached_value(self):
        if None != self.logging_level:
            logging.log(
                self.logging_level, 'Memoize: %s: clearing cached value.', self._callable.__name__)
        try:
            del self.cached_value
        except AttributeError:
            pass


memoize = simple_decorator(Memoize)


@simple_decorator
def log_result(fn):
    "Decorator to log the result of a function."
    def wrapper(*args, **kwds):
        result = fn(*args, **kwds)
        logging.info("Result of %s: %s", fn.__name__, str(result))
        return result
    return wrapper
