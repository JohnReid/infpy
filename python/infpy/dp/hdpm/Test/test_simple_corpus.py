#
# Copyright John Reid 2010
#


"""
Code to test HDPM with a simple corpus.

The corpus consists of 2 documents over a vocabulary of 2 words. The first document consists entirely of the
first word and the second the second word. We test that a HDPM only needs 2 programs at most to model the corpus.
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


for model_idx in range(options.num_models_to_test):
    LL, model = infer_model(hdpm.HDPM(documents, W, K), options)

    # Check we explained the data using a small number of topics.
    # We want the number of words explained by the first W topics to be at least 99% of all the words.
    if model.counts.E_n_dk.sum(axis=0)[:W].sum() >= .99 * model.N:
        logging.info(
            'Found a model that could explain the corpus using two topics.')
        summarise_model(model, LL)
        break
else:
    raise RuntimeError(
        'Could not infer model that explained the corpus using two topics.')
