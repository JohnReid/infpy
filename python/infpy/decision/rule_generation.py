#
# Copyright John Reid 2007
#

"""
Implements rules to generate and modify decision trees
"""

from attributes import *

from random import gauss, random, sample, randint, choice

class Context(object):
    """
    The context object defines the data's attributes and their range of values.
    It also defines the possible outcomes (classifications) of a decision tree.
    """
    attributes = []
    outcomes = []


class Rule(object):
    """
    A rule in a decision tree.

    A rule uses one attribute and to choose between several possible outcomes
    """
    def __init__(self, attr): self.attr = attr
    def __deepcopy__(self, memo): raise RuntimeError('Subclasses should implement __deepcopy__')



class StraightThroughRule(Rule):
    """
    A rule that simply returns the attribute's value
    """
    def __init__(self, attr): Rule.__init__(self, attr)
    def __call__(self, data): return self.attr.fn(data)


class ComparisonRule(Rule):
    """
    A rule that compares the attribute's value with a constant
    """
    def __init__(self, attr, constant):
        Rule.__init__(self, attr)
        self.constant = constant
    def __call__(self, data):
        return self.attr.fn(data) > self.constant and 1 or 0
    def __str__(self):
        return '%s > %f' % (self.attr.name, self.constant)
    def __deepcopy__(self, memo):
        return ComparisonRule(self.attr, self.constant)

class MembershipRule(Rule):
    """
    A rule that tests whether the attribute's value is a member of some set
    """
    def __init__(self, attr, iterable):
        Rule.__init__(self, attr)
        self.set = set(iterable)
    def __call__(self, data):
        return self.attr.fn(data) in self.set and 1 or 0
    def __str__(self):
        return '%s in {%s}' % (self.attr.name, ",".join(str(v) for v in self.set))
    def __deepcopy__(self, memo):
        return MembershipRule(self.attr, self.set)



def new_rule_for_attribute(attr):
    """
    Creates a new rule based on the attribute

    Returns (rule,# possible outcomes)

    A rule is function object that takes one argument (a datum) and returns
    an index into some children
    """
    if isinstance(attr, EnumerativeAttribute):
        return MembershipRule(attr, sample(xrange(attr.num_values),attr.num_values/2)), 2

    elif isinstance(attr, OrdinalAttribute):
        return ComparisonRule(attr, randint(0, attr.num_values - 2)), 2

    elif isinstance(attr, ContinuousAttribute):
        return ComparisonRule(attr, gauss(attr.mean, attr.stddev)), 2

    else: raise RuntimeError('Unknown attribute type')

    raise RuntimeError('Creating rules for %s attributes not implemented' % attr.__class__)


def new_rule(attributes):
    """
    Chooses one of the attributes and creates a rule based on it.

    See new_rule_for_attribute
    """
    return new_rule_for_attribute(choice(attributes))



def new_mutated_rule(rule, degree_of_mutation = 1.0):
    "Create a new mutated copy of a rule"
    from random import random
    if isinstance(rule.attr, EnumerativeAttribute):
        # randomly remove/insert elements from rule set
        p_mutation = degree_of_mutation / rule.attr.num_values
        s = set()
        for value in xrange(rule.attr.num_values):
            if (random() > p_mutation) ^ (value in rule.set): s.add(value)
        return MembershipRule(rule.attr, s)

    elif isinstance(rule.attr, OrdinalAttribute):
        # add or take away a little from the comparision constant
        if random() > .5: constant = rule.constant + int(degree_of_mutation)
        else: constant = rule.constant - int(degree_of_mutation)
        constant = min(constant, rule.attr.num_values - 2)
        constant = max(constant, 0)
        return ComparisonRule(rule.attr, constant)

    elif isinstance(rule.attr, ContinuousAttribute):
        # move towards another sample from the distribution
        new_sample = gauss(rule.attr.mean, rule.attr.stddev)
        return ComparisonRule(
                rule.attr,
                (degree_of_mutation * new_sample + rule.constant) / (1.0 + degree_of_mutation)
        )

    else: raise RuntimeError('Cannot mutate rules with this attribute type: ' + rule.attr.__class__)


def only_possible_outcome(satisfied_rule, outcome, rule_to_test):
    """
    Given that the satisfied_rule has the given outcome, are we guaranteed what
    the outcome of the rule_to_test is?

    Returns the outcome if we are sure, returns None if not
    """

    # cannot be sure with different attributes
    if satisfied_rule.attr != rule_to_test.attr: return None

    if isinstance(satisfied_rule, ComparisonRule) and isinstance(rule_to_test, ComparisonRule):
        if 1 == outcome:
            if satisfied_rule.constant > rule_to_test.constant:
                return outcome
        else:
            if satisfied_rule.constant < rule_to_test.constant:
                return outcome

    elif isinstance(satisfied_rule, MembershipRule) and isinstance(rule_to_test, MembershipRule):
        if 1 == outcome:
            if satisfied_rule.set.issuperset(rule_to_test.set):
                return outcome
        else:
            if satisfied_rule.set.issubset(rule_to_test.set):
                return outcome
