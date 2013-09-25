#
# Copyright John Reid 2010
#


"""
Code to examine the output from a profile run, e.g. check_performance.py.
"""


import pstats


p = pstats.Stats('hdpm.prof')
p.sort_stats('cumulative').print_stats(30)

# p.sort_stats('time').print_stats(30)
