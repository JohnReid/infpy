#!/bin/bash

PYTHONPATH=.:$PYTHONPATH

python2.5 test_dimensions.py || exit -1
# python2.5 test_hdpm.py || exit -1
python2.5 test_hyperparameters.py -t 1e-4 -I 500 || exit -1
python2.5 test_LL.py || exit -1
python2.5 test_simple_corpus.py || exit -1
# python2.5 test_update_order.py || exit -1
python2.5 test_without_memoisation.py || exit -1
