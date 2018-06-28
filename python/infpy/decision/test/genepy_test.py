#
# Copyright John Reid 2007, 2010
#


import unittest
import logging
from infpy.genepy import Population
from infpy.decision import Context, OrdinalAttribute, EnumerativeAttribute, ContinuousAttribute, DecisionTreeSpecies, count_nodes, log_tree


def ord(data): return data[0]


def enum(data): return data[1]


def cont(data): return data[2]


class GenepyTest(unittest.TestCase):
    """Test case for decision tree genepy interface"""

    # define the context - i.e. what attributes the data have and their
    # classification
    context = Context()
    context.attributes.append(OrdinalAttribute('ordinal', ord, 10))
    context.attributes.append(EnumerativeAttribute('enumerative', enum, 4))
    context.attributes.append(
        ContinuousAttribute('continuous', cont, 0.0, 1.0))
    context.outcomes = [0, 1, 2]

    # generate some data
    from random import gauss, randint
    data = [
        ((randint(0, 9), randint(0, 3), gauss(0.0, 1.0)), randint(0, 2))
        for i in xrange(30)
    ]

    species = DecisionTreeSpecies(context)
    species.data = data

    def fitness_fn(self, individual):
        loss = 0
        for x, y in self.data:
            if y != individual(x):
                loss += 1
        return loss

    def test_initialisation(self):
        self.species.random_individual()

    def test_mutation(self):
        pass  # self.species.mutate(self.species.random_individual())

    def test_combination(self):
        self.species.mate(self.species.random_individual(),
                          self.species.random_individual())

    def test_call(self):
        for x, _y in self.data:
            self.species.random_individual()(x)

    def test_overall(self):
        pop = Population(size=100, species=self.species,
                         fitness_fn=self.fitness_fn)
        pop.post_generation_process = self.species.prune_individuals
        for gen_idx in xrange(100):
            # execute a generation
            pop.generation()

            # and print how good best is
            tree_sizes = list(count_nodes(i) for i in pop.individuals)
            logging.debug("%3d; best fitness=%2d; average tree size: %3.3f", gen_idx, int(
                pop.most_fit.fitness), float(sum(tree_sizes)) / len(tree_sizes))
        log_tree(pop.most_fit, logging.getLogger(), level=logging.DEBUG)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
