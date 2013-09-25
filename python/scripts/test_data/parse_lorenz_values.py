#
# Copyright John Reid 2010
#

"""
Parse Lorenz's p-values and write in format suitable for fit beta mixture script.
"""

import csv

lorenz_file = 'lorenz-p-values.csv'

# create file handles to put output in
outputs = (
    open('cshl-p-values.txt', 'w'),
    open('sav-p-values.txt', 'w'),
    open('lorenz-weights.txt', 'w'),
)
inv_weights = open('lorenz-inverse-weights.txt', 'w')

# create CSV parser
reader = csv.reader(open(lorenz_file))

# ignore first line
reader.next()

# for each row
biggest = 0.
smallest = 1.
eps = 1e-16
for row in reader:
    if 3 != len(row):
        raise ValueError('Wrong number of fields in row')
    values = map(float, row)
    for f, v in zip(outputs, values):
        if v < 1. and biggest < v:
            biggest = v
        if v > 0. and smallest > v:
            smallest = v
        if v < eps:
            v = eps
        elif v > 1. - eps:
            v = 1. - eps
        f.write('%.20e\n' % v)
    print >> inv_weights, 1. - float(row[2])

# close file handles
for f in outputs:
    f.close()
inv_weights.close()

print 'p-values range from %20g to 1 - %20g' % (smallest, 1.-biggest)
