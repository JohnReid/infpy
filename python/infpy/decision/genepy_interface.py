#
# Copyright John Reid 2007, 2010
#

"""
Implements decision tree interface to the genepy genetic algorithm library
"""

from decision import create_random_decision_tree, combine_trees, mutate_tree, prune_tree, print_tree


class DecisionTreeSpecies(object):
    """
    Wraps a decision tree to interface to the pygene genetic algorithm library

    Implements combination, mutation and random initialisation
    """

    def __init__(
            self,
            context,
            initial_depth=3,
            p_mutation=.1
    ):
        self.context = context
        self.initial_depth = initial_depth
        self.p_mutation = p_mutation

    def random_individual(self):
        "Random initialisation"
        return create_random_decision_tree(self.context, self.initial_depth)

    def mate(self, individual_1, individual_2):
        "Recmbination"
        from copy import deepcopy
        tree_1 = deepcopy(individual_1)
        tree_2 = deepcopy(individual_2)
        combine_trees(tree_1, tree_2)
        return tree_1

    def mutate(self, individual):
        "Mutation in place"
        mutate_tree(individual, self.context, 4)

    def prune_individuals(self, pop):
        from random import random
        if random() < .1:
            check_pruning = False  # if true check pruning does not change results on self.data
            for i in pop.individuals:
                if check_pruning:
                    pre_pruning_outcomes = [i(x) for x, _y in self.data]
                from copy import deepcopy
                pre_i = deepcopy(i)
            prune_tree(i)
            if check_pruning:
                post_pruning_outcomes = [i(x) for x, _y in self.data]
                for pre, post in zip(pre_pruning_outcomes, post_pruning_outcomes):
                    if pre != post:
                        print_tree(pre_i)
                        print_tree(i)
                        print '%d %d' % (pre, post)
