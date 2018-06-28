#
# Copyright John Reid 2007, 2010
#

"""
Implementation of a decision tree
"""

from .rule_generation import new_rule, new_mutated_rule, only_possible_outcome
from random import choice, shuffle, random, randint


class DecisionNode(object):
    """
    A node in a decision tree that makes a decision based on the data
    """

    def __init__(self, rule, children):
        "When the rule is evaluated on data it returns an index into the children"
        self.rule = rule
        self.children = children

    def __call__(self, data):
        """
        Evaluates the decision tree on the given data, i.e. calls the rule and then
        recurses into the selected child node
        """
        return self.children[self.rule(data)](data)

    def __str__(self):
        return 'if %s:' % self.rule


class LeafNode(object):
    """
    A node in a decision tree that returns one of the possible outcomes
    """

    def __init__(self, outcome):
        self.outcome = outcome

    def __call__(self, data):
        """
        Evaluates the decision tree on the given data, i.e. returns outcome for this
        leaf node
        """
        return self.outcome

    def __str__(self):
        return str(self.outcome)


def print_tree(tree, indent=0):
    print('%s%s' % (' ' * indent, str(tree)))
    if isinstance(tree, DecisionNode):
        for c in tree.children:
            print_tree(c, indent + 1)
    elif isinstance(tree, LeafNode):
        pass
    else:
        raise RuntimeError(
            'Cannot print tree nodes of this type: ' + tree.__class__)


def log_tree(tree, logger, level, indent=0):
    logger.log(level, '%s%s', ' ' * indent, str(tree))
    if isinstance(tree, DecisionNode):
        for c in tree.children:
            log_tree(c, logger, level, indent=indent + 1)
    elif isinstance(tree, LeafNode):
        pass
    else:
        raise RuntimeError(
            'Cannot print tree nodes of this type: ' + tree.__class__)


class NodeCounter(object):
    count = 0

    def __call__(self, node): self.count += 1


def count_nodes(tree):
    return visit_tree_nodes(tree, NodeCounter()).count


def visit_tree_nodes(tree, visitor):
    if None != tree:
        visitor(tree)
        if isinstance(tree, DecisionNode):
            for c in tree.children:
                visit_tree_nodes(c, visitor)
    return visitor


def create_random_decision_tree(context, depth):
    if 0 == depth:
        return LeafNode(choice(context.outcomes))
    else:
        rule, num_outcomes = new_rule(context.attributes)
        children = [
            create_random_decision_tree(context, depth - 1) for i in range(num_outcomes)
        ]
        return DecisionNode(rule, children)


def replace_rule(decision_node, context, depth=0):
    "Replace the rule in the node with a completely new random rule"
    decision_node.rule, num_outcomes = new_rule(context.attributes)
    while num_outcomes > len(decision_node.children):  # add children if needed
        decision_node.children.append(
            create_random_decision_tree(context, depth))
    while num_outcomes < len(decision_node.children):  # remove children if too many
        decision_node.children.pop()
    shuffle(decision_node.children)  # randomise order


def insert_decision(decision_node, context, depth=0):
    "Insert an extra decision in the tree"
    child_to_replace = choice(decision_node.children)
    idx_to_replace = decision_node.children.index(child_to_replace)
    rule, num_outcomes = new_rule(context.attributes)
    children = [child_to_replace]
    while num_outcomes > len(children):  # add other children
        children.append(create_random_decision_tree(context, depth))
    decision_node.children[idx_to_replace] = DecisionNode(rule, children)


def mutate_node(node, context):
    "Mutate the given node"
    if isinstance(node, DecisionNode):
        p = random()
        if p < .3:
            replace_rule(node, context, 1)
        elif p < .35:
            insert_decision(node, context, 1)
        else:
            node.rule = new_mutated_rule(node.rule)
    elif isinstance(node, LeafNode):
        node.outcome = choice(context.outcomes)
    else:
        raise RuntimeError(
            'Cannot mutate nodes of this type: ' + node.__class__)


class NodeMutator(object):
    "Mutates nodes according to some probability"

    def __init__(self, p_mutation, context):
        self.p_mutation = p_mutation
        self.context = context

    def __call__(self, node):
        if random() < self.p_mutation:
            mutate_node(node, self.context)


def build_random_path(tree, path):
    "Build a random path in tree to a leaf node"
    path.append(tree)
    if isinstance(tree, DecisionNode):
        build_random_path(choice(tree.children), path)
    elif isinstance(tree, LeafNode):
        pass
    else:
        raise RuntimeError(
            'Cannot build random path over nodes of this type: ' + tree.__class__)


def choose_random_decision_node(tree):
    "Returns (parent,node) where parent is always a DecisionNode"
    random_path = []
    build_random_path(tree, random_path)  # get the decision nodes
    if len(random_path) < 2:
        return (None, None)
    else:
        idx = randint(0, len(random_path) - 2)  # don't choose last leaf node
        return random_path[idx], random_path[idx + 1]


def combine_trees(tree_1, tree_2):
    "Randomly combine 2 trees in place"

    # find 2 nodes to swap in the trees
    parent_1, node_1 = choose_random_decision_node(tree_1)
    parent_2, node_2 = choose_random_decision_node(tree_2)
    if None == parent_1 or None == parent_2:
        return False  # don't do anything if we can't

    # swap
    idx1 = parent_1.children.index(node_1)
    idx2 = parent_2.children.index(node_2)
    parent_1.children[idx1] = node_2
    parent_2.children[idx2] = node_1


def mutate_tree(tree, context, new_subtree_depth):
    "Replace one node with a new random subtree of the given depth"
    parent, node = choose_random_decision_node(tree)
    if None != parent and None != node:
        idx = parent.children.index(node)
        parent.children[idx] = create_random_decision_tree(
            context, new_subtree_depth)


class PossibleOutcomes(object):
    s = set()

    def __call__(self, node):
        if isinstance(node, LeafNode):
            self.s.add(node.outcome)


def tree_possible_outcomes(tree):
    "Returns all the possible outcomes of this tree"
    return visit_tree_nodes(tree, PossibleOutcomes()).s


def tree_has_only_one_outcome(tree, outcome=None):
    """
    Does this tree have only one outcome and what is it?

    Returns None if more than one outcome, otherwise returns outcome
    """
    if isinstance(tree, LeafNode):
        if None == outcome:
            outcome = tree.outcome
        else:
            if outcome != tree.outcome:
                return None
    elif isinstance(tree, DecisionNode):
        for c in tree.children:
            if None == tree_has_only_one_outcome(c, outcome):
                return None
        return outcome
    else:
        raise RuntimeError(
            'Cannot find outcome of nodes of this type: ' + tree.__class__)


def prune_tree(tree, rules_satisfied=None):
    "Prunes the tree to remove useless nodes"
    if None == rules_satisfied:
        rules_satisfied = []
    if isinstance(tree, DecisionNode):
        for i, c in enumerate(tree.children):
            outcome = tree_has_only_one_outcome(c)
            if None != outcome:
                tree.children[i] = LeafNode(outcome)
            else:
                rules_satisfied.append((tree.rule, i))
                if isinstance(c, DecisionNode):
                    for rule, outcome in rules_satisfied:
                        only_outcome = only_possible_outcome(
                            rule, outcome, c.rule)
                        if None != only_outcome:
                            # remove this level of decision making
                            tree.children[i] = c.children[only_outcome]
                prune_tree(c, rules_satisfied)
                rules_satisfied.pop()
