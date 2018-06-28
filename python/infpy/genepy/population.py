#
# Copyright John Reid 2007
#

from random import random
from itertools import chain


class Population(object):
    """
    A collection of individuals of a particular species
    """

    species = None  # defines initialisation, reproduction, and mutation
    fitness_fn = None  # scores an individual
    minimise = True  # minimise or maximise the fitness?
    breeding_proportion = .1  # proportion of population selected as parents
    carry_over_parents = True  # include parents in next generation?
    p_mutation = .1  # chance of mutation of an individual at each generation
    post_generation_process = None  # function to be called at end of every generation

    def __init__(
            self,
            size,
            species,
            fitness_fn
    ):
        self.species = species
        self.fitness_fn = fitness_fn
        self.individuals = [self.species.random_individual()
                            for i in range(size)]

    def generation(self):
        "Run through a generation"
        self.score_individuals()
        self.select_parents()
        self.reproduce()
        self.combine_generations()
        self.mutate()
        if self.post_generation_process:
            self.post_generation_process(self)

    def score_individuals(self):
        for i in self.individuals:
            i.fitness = self.fitness_fn(i)

    def select_parents(self):
        self.individuals.sort(key=lambda x: x.fitness,
                              reverse=not self.minimise)
        self.most_fit = self.individuals[0]
        num_parents = int(self.breeding_proportion * len(self.individuals))
        self.parents = self.individuals[:num_parents]

    def reproduce(self):
        # how many children
        if self.carry_over_parents:
            num_children = len(self.individuals) - len(self.parents)
        else:
            num_children = len(self.individuals)

        from random import choice
        self.children = [self.species.mate(choice(self.parents), choice(
            self.parents)) for i in range(num_children)]

    def combine_generations(self):
        self.individuals = list(chain(self.parents, self.children))

    def mutate(self):
        for i in self.individuals:
            if random() < self.p_mutation:
                self.species.mutate(i)
