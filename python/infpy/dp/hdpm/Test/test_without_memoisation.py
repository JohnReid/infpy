#
# Copyright John Reid 2010
#


"""
Code to test the HDPM still works when we turn off caching of intermediary results.
"""

from simple_corpus import *

logging.basicConfig(level=logging.INFO)

parser = create_option_parser()
options, args = parse_options(parser)



def create_seeded_model():
    numpy.random.seed(options.seed)
    return create_model()

def update_model(model):
    model.update()
    model.update()
    return model.log_likelihood()


#
# Check when turning all cached functions off
#
model = create_seeded_model()
for cachable_fn in model._cachable_fns():
    cachable_fn.set_cache_enabled(False)
LL_with_caching = update_model(model)

model = create_seeded_model()
model._cachable_fns()[i].set_cache_enabled(False)
LL_without_caching = update_model(model)

assert LL_with_caching == LL_without_caching



#
# Check when turning each cached function off individually
#
num_cachable_fns = len(create_model()._cachable_fns())
for i in xrange(num_cachable_fns):
    model = create_seeded_model()
    model._cachable_fns()[i].set_cache_enabled(False)
    LL_with_caching = update_model(model)

    model = create_seeded_model()
    model._cachable_fns()[i].set_cache_enabled(False)
    LL_without_caching = update_model(model)

    assert LL_with_caching == LL_without_caching
