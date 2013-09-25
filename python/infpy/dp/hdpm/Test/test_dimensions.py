#
# Copyright John Reid 2010
#


"""
Code to test HDPMs with different Ds, Ks and Ws to hopefully spot any bugs where matrices are indexed by the wrong dimension.
"""

import infpy.dp.gen_artificial_data as gen_artificial_data, numpy, logging
import infpy.dp.hdpm as hdpm

logging.basicConfig(level=logging.INFO)


def create_model(D, d, K, W):
    tau = numpy.ones((W,))

    sampler = gen_artificial_data.HDPMSampler(
        numpy.ones((D,), dtype=int) * d, # document sizes
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


numpy.random.seed(1)

for D, d, K, W in (
    (  2, 100,   5,   2),
    (  2, 100,  13,  40),
    ( 20,  10,   5,   2),
    ( 10,  50,  50, 200),
    (500,   2,   9,  20),
):
    logging.info('D=%3d; d=%3d; K=%3d; W=%3d', D, d, K, W)
    model = create_model(D, d, K, W)
    for i in xrange(3):
        model.update()
