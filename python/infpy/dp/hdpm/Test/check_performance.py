#
# Copyright John Reid 2010
#


"""
Code to test the performance of the HDPM. Use with the cProfile module:

python -O -m cProfile -o hdpm.prof Test/check_performance.py
"""

import gen_artificial_data
import hdpm
import numpy
import logging
from optparse import OptionParser

logging.basicConfig(level=logging.INFO)

if __debug__:
    raise RuntimeError("""
    Python __debug__ variable is True.
    Please run this script with the -O flag.
    This will turn off assert statements.
""")

parser = OptionParser()
parser.add_option(
    "-s",
    dest="seed",
    default=1,
    type='int',
    help="Seed for RNG."
)
parser.add_option(
    "-I",
    dest="iterations",
    type='int',
    default=50,
    help="Iterations to run."
)

options, args = parser.parse_args()
logging.info('Options:')
for option in parser.option_list:
    if option.dest:
        logging.info('%32s: %-32s * %s', option.dest,
                     str(getattr(options, option.dest)), option.help)

numpy.random.seed(options.seed)


def create_model(D, d, K, W):
    tau = numpy.ones((W,))

    sampler = gen_artificial_data.HDPMSampler(
        numpy.ones((D,), dtype=int) * d,  # document sizes
        1., 1.,
        1., 1.,
        1., 1.,
        tau
    )

    model = hdpm.HDPM(
        [document[2] for document in sampler.documents],
        W,
        K,
        1., 1.,
        1., 1.,
        1., 1.,
        tau
    )

    return model


for D, d, K, W in (
    (100, 30, 100, 60),
    #    (  2, 100,   5,   2),
    #    (  2, 100,  13,  40),
    #    ( 20,  10,   5,   2),
    #    ( 10,  50,  50, 200),
    #    (500,   2,   9,  20),
):
    logging.info(
        'Testing model of dimensions: D=%3d; d=%3d; K=%3d; W=%3d', D, d, K, W)
    model = create_model(D, d, K, W)
    for i in range(options.iterations):
        model.update()
