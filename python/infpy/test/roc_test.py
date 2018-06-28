#
# Copyright John Reid 2013
#

"""
Code to test ROC point/curve calculation and plotting.
"""

from .. import roc

perfect_positive = [0.]
perfect_negative = [.1, .2, .3, .4]


def pairs(iterable):
    """Yield all pairs from the iterable.

    I.e. [0,1,2,3,4] yields:
    (0,1)
    (1,2)
    (2,3)
    (3,4)
    """
    i = iter(iterable)
    try:
        last = next(i)
        while True:
            this = next(i)
            yield last, this
            last = this
    except StopIteration:
        pass


def test_pairs():
    iterable = [0, 1, 2, 3, 4]
    p = list(pairs(iterable))
    assert len(p) == len(iterable) - 1
    assert p[0] == (0, 1)
    assert p[1] == (1, 2)
    assert p[2] == (2, 3)
    assert p[3] == (3, 4)


def check_roc_invariants(rocs, include_endpoints=True):
    if include_endpoints:
        print(rocs[0])
        assert 0. == rocs[0].sensitivity(), '0 != %.f' % rocs[0].sensitivity()
        assert 1. == rocs[0].specificity(), '1 != %.f' % rocs[0].specificity()
        print()
        print(rocs[-1])
        assert 1. == rocs[-1].sensitivity(), '1 != %.f' % rocs[-1].sensitivity()
        assert 0. == rocs[-1].specificity(), '0 != %.f' % rocs[-1].specificity()
    for roc1, roc2 in pairs(rocs):
        print()
        print(roc1)
        print(roc2)
        assert roc1.sensitivity() <= roc2.sensitivity()
        assert roc1.specificity() >= roc2.specificity()


def test_all_rocs():
    rocs = list(roc.all_rocs_from_thresholds(
        perfect_positive, perfect_negative))
    check_roc_invariants(rocs[::-1])


def test_create_rocs():
    rocs = list(roc.create_rocs_from_thresholds(
        perfect_positive, perfect_negative))
    check_roc_invariants([r for r, t in rocs], include_endpoints=False)


def test_rocs():
    rocs = list(roc.rocs_from_thresholds(perfect_positive, perfect_negative))
    check_roc_invariants(rocs, include_endpoints=False)
