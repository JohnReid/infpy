#
# Copyright John Reid 2010
#


"""
Code to test hyperparameters of a HDPM with a simple corpus.

The corpus consists of 2 documents over a vocabulary of 2 words. The first document consists entirely of the
first word and the second the second word. We test that changing the hyper-parameters can affect if the HDPM
learns one program that is a mixture of the 2 words or 2 programs consisting of one word each.
"""

from simple_corpus import *

logging.basicConfig(level=logging.INFO)

parser = create_option_parser()
parser.add_option(
    "-n",
    dest="num_models_to_test",
    default=10,
    type='int',
    help="Number of models to test."
)
options, args = parse_options(parser)
numpy.random.seed(options.seed)


#
# Now manipulate hyperparameters of HDPM so that it explains data using one topic
#
logging.info(
    'Trying hyperparameters that should make HDPM explain data using one topic.')

#
# If alpha is large, then all the documents are more likely to have similar topic distributions
# If beta is large then all the topics should have similar word distributions
# If gamma is small then we should have few partitions
# If a_tau is large then tau should be very similar to it
#
a_alpha, b_alpha = 10., .1
# beta  should be large to encourage all the topics to have similar word distributions
a_beta, b_beta  = 10., .1
a_gamma, b_gamma = .1, 10.  # gamma should be small in order to encourage few partitions
a_tau = numpy.ones(W) * 1e2

for model_idx in range(options.num_models_to_test):
    model = hdpm.HDPM(
        documents, W, K,
        a_alpha, b_alpha,
        a_beta, b_beta,
        a_gamma, b_gamma,
        a_tau
    )

    LL, model = infer_model(model, options)

    # Check we explained the data using one topic.
    if model.counts.E_n_k[0] >= .99 * model.N:
        logging.info('Found a model that explains the corpus using one topic.')
        summarise_model(model, LL)
        break
else:
    raise RuntimeError("""
Could not infer model that explained the corpus using one topic. Perhaps you should run the
test again with a smaller tolerance and more iterations, e.g. -t 1e-4 -I 500
""")
