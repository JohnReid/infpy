#!/bin/bash 

PYTHONPATH=.:$PYTHONPATH

REGEX="INFO:root:LL converged to"
REGEX="E_s_dk: "
REGEX="_calculate_E_s_dk: "

echo "Running with disabled cache"
python2.5 Test/test_without_memoisation.py -I 1 --dis 2>&1 | tee output-disabled.log | grep "$REGEX" | tee grepped-disabled.log

echo "Running without disabled cache"
python2.5 Test/test_without_memoisation.py -I 1 2>&1 | tee output-enabled.log | grep "$REGEX" | tee grepped-enabled.log

