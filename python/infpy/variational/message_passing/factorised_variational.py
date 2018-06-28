#
# Copyright John Reid 2008
#

"""
Code for variational message passing.
"""

import boost.graph as bgl
import numpy.random as R
import os
import numpy
import scipy.special
import math
from infpy.convergence_test import LlConvergenceTest


if __debug__:
    def assert_result_finite(fn):
        "A decorator that asserts the result of the function is finite."
        def new(*args):
            result = fn(*args)
            if not numpy.isfinite(result).all():
                raise AssertionError
            return result
        return new
else:
    def assert_result_finite(fn):
        "A decorator that does nothing."
        return fn


class FactorGraph(object):
    "A factor graph."

    def __init__(self, g, variables, factors):
        self.g = g
        "The factor graph."

        self.variables = variables
        "Maps graph vertices to variables."

        self.factors = factors
        "Maps graph vertices to factors."

    def add_variable(self, variable):
        "@return: The vertex added."
        v = self.g.add_vertex()
        self.variables[v] = variable
        return v

    def add_factor(self, factor):
        "@return: The vertex added."
        v = self.g.add_vertex()
        self.factors[v] = factor
        factor.connect(g, v)
        return v

    def factor_vertices(self):
        "@return: The vertices that represent factors."
        return [v for v in g.vertices if None != factors[v]]

    def variable_vertices(self):
        "@return: The vertices that represent variables."
        return [v for v in g.vertices if None != variables[v]]


class VarDist(object):
    "A variational distribution."
    pass


class VarDistObserved(VarDist):
    """
    A variational distribution for an observed value. This is really just a delta distribution on the
    observed value
    """

    def __init__(self, observed_value):
        self.observed_value = observed_value

    def update(self, expectations):
        pass

    @assert_result_finite
    def get_expectation(self):
        return self.observed_value

    def entropy(self):
        "@return: The entropy of this variational distribution."
        return 0.

    def accept(self, visitor, *args, **kwargs):
        """
        Accept a visitor according to the
        U{visitor design pattern<http://en.wikipedia.org/wiki/Visitor_pattern>}.
        """
        return visitor.visit_var_dist_observed(self, *args, **kwargs)


class VarDistHidden(VarDist):
    "A variational distribution."

    def update(self, expectations):
        assert numpy.isfinite(expectations).all()
        self.parameters[:] = expectations

    def accept(self, visitor, *args, **kwargs):
        """
        Accept a visitor according to the
        U{visitor design pattern<http://en.wikipedia.org/wiki/Visitor_pattern>}.
        """
        return visitor.visit_var_dist_hidden(self, *args, **kwargs)


class VarDistGaussian(VarDistHidden):

    def __init__(self):
        self.parameters = numpy.array([0., -.5])

    def entropy(self):
        "@return: The entropy of this variational distribution."
        mu, gamma = FactorGaussian.extract_parameters(self.parameters)
        return .5 * (1. + FactorGaussian._log_2_pi - numpy.log(gamma))

    @assert_result_finite
    def get_expectation(self):
        mu, gamma = FactorGaussian.extract_parameters(self.parameters)
        return numpy.array([mu, 1. / gamma + mu**2])


class VarDistGamma(VarDistHidden):
    def __init__(self):
        self.parameters = numpy.array([-1., 0.])

    def entropy(self):
        "@return: The entropy of this variational distribution."
        a, b = FactorGamma.extract_parameters(self.parameters)
        return (
            a
            + (1. - a) * scipy.special.digamma(a)
            - numpy.log(b)
            + scipy.special.gammaln(a)
        )

    @assert_result_finite
    def get_expectation(self):
        a, b = FactorGamma.extract_parameters(self.parameters)
        return numpy.array([a / b, scipy.special.digamma(a) - numpy.log(b)])


class VarDistDiscrete(VarDistHidden):
    def __init__(self, K=2):
        self.parameters = numpy.ones(K) / K

    def update(self, expectations):
        assert numpy.isfinite(expectations).all()
        self.parameters[:] = numpy.exp(expectations)
        self.parameters /= self.parameters.sum()

    def entropy(self):
        "@return: The entropy of this variational distribution."
        return -numpy.dot(numpy.log(self.parameters), self.parameters)

    @assert_result_finite
    def get_expectation(self):
        return self.parameters


class Variable(object):
    pass


class Factor(object):
    def connect(self, g, v):
        "Connect this factor to its neighbours (i.e. variables)"
        for u in self.neighbours():
            g.add_edge(u, v)


class FactorGaussian(Factor):
    """
    A factor in a factor graph that represents a Gaussian distribution.
    """

    _log_2_pi = math.log(2 * math.pi)

    def accept(self, visitor, *args, **kwargs):
        """
        Accept a visitor according to the
        U{visitor design pattern<http://en.wikipedia.org/wiki/Visitor_pattern>}.
        """
        return visitor.visit_factor_gaussian(self, *args, **kwargs)

    def __init__(self, x, mu, gamma):
        """
        Set which variables represent x, mu and gamma in this Gaussian distribution.

        @arg x: The vertex representing x.
        @arg mu: The vertex representing mu.
        @arg gamma: The vertex representing gamma.
        """
        self.x = x
        self.mu = mu
        self.gamma = gamma

    def neighbours(self):
        "@return: The vertices of this factor's variables"
        return [self.x, self.mu, self.gamma]

    @assert_result_finite
    def log_likelihood(self, var_dists):
        "@return: The log likelihood of this factor under the variational distribution."
        x, x2 = var_dists[self.x].get_expectation()
        mu, mu2 = var_dists[self.mu].get_expectation()
        gamma, log_gamma = var_dists[self.gamma].get_expectation()
        return (
            gamma * mu * x
            + .5 * (
                log_gamma
                - gamma * mu2
                - FactorGaussian._log_2_pi
                - gamma * x2
            )
        )

    @assert_result_finite
    def calculate_update_for(self, v, var_dists):
        """
        Get an update for the variable whose vertex is v.
        """
        if self.x == v:
            mu, mu2 = var_dists[self.mu].get_expectation()
            gamma, log_gamma = var_dists[self.gamma].get_expectation()
            return numpy.array([
                mu * gamma,
                -.5 * gamma
            ])
        if self.mu == v:
            x, x2 = var_dists[self.x].get_expectation()
            gamma, log_gamma = var_dists[self.gamma].get_expectation()
            return numpy.array([
                gamma * x,
                -.5 * gamma
            ])
        if self.gamma == v:
            x, x2 = var_dists[self.x].get_expectation()
            mu, mu2 = var_dists[self.mu].get_expectation()
            return numpy.array([
                mu * x - .5 * (x2 + mu2),
                .5
            ])
        raise RuntimeError(
            'Neighbour, v, not known by factor. Cannot calculate an update for unknown vertex.')

    @staticmethod
    def natural_x(x):
        "Convert from canonical to natural form."
        return numpy.array((x, x**2))

    @staticmethod
    def natural_mu(mu):
        "Convert from canonical to natural form."
        return numpy.array((mu, mu**2))

    @staticmethod
    def natural_gamma(gamma):
        "Convert from canonical to natural form."
        return numpy.array((gamma, numpy.log(gamma)))

    @staticmethod
    def extract_parameters(parameters):
        gamma = -2. * parameters[1]
        mu = parameters[0] / gamma
        return mu, gamma


class FactorGamma(Factor):
    """
    A factor in a factor graph that represents a Gamma distribution.
    """

    def accept(self, visitor, *args, **kwargs):
        """
        Accept a visitor according to the
        U{visitor design pattern<http://en.wikipedia.org/wiki/Visitor_pattern>}.
        """
        return visitor.visit_factor_gamma(self, *args, **kwargs)

    def __init__(self, x, a, b):
        """
        Set which variables represent x, a and b in this Gamma distribution.

        @arg x: The vertex representing x.
        @arg a: The vertex representing a.
        @arg b: The vertex representing b.
        """
        self.x = x
        self.a = a
        self.b = b

    def neighbours(self):
        "@return: The vertices of this factor's variables"
        return [self.x, self.a, self.b]

    @assert_result_finite
    def calculate_update_for(self, v, var_dists):
        """
        Get an update for the variable whose vertex is v.
        """
        if self.x == v:
            a, log_gamma_a = var_dists[self.a].get_expectation()
            b, log_b = var_dists[self.b].get_expectation()
            return numpy.array([
                -b,
                a - 1.
            ])
        if self.a == v:
            x, log_x = var_dists[self.x].get_expectation()
            b, log_b = var_dists[self.b].get_expectation()
            return numpy.array([
                log_b + log_x,
                -1.
            ])
        if self.b == v:
            x, log_x = var_dists[self.x].get_expectation()
            a, log_gamma_a = var_dists[self.a].get_expectation()
            return numpy.array([
                -x,
                a
            ])
        raise RuntimeError(
            'Neighbour, v, not known by factor. Cannot calculate an update for unknown vertex.')

    @assert_result_finite
    def log_likelihood(self, var_dists):
        "@return: The log likelihood of this factor under the variational distribution."
        x, log_x = var_dists[self.x].get_expectation()
        a, log_gamma_a = var_dists[self.a].get_expectation()
        b, log_b = var_dists[self.b].get_expectation()
        return (
            -b * x
            + (a - 1.) * log_x
            + a * log_b
            - log_gamma_a
        )

    @staticmethod
    def natural_x(x):
        "Convert to natural form."
        return numpy.array((x, numpy.log(x)))

    @staticmethod
    def natural_a(a):
        "Convert to natural form."
        return numpy.array((a, scipy.special.gammaln(a)))

    @staticmethod
    def natural_b(b):
        "Convert to natural form."
        return numpy.array((b, numpy.log(b)))

    @staticmethod
    def extract_parameters(parameters):
        b = -parameters[0]
        a = parameters[1] + 1.
        return a, b


class FactorMixture(Factor):
    """
    A factor in a factor graph that models a mixture of several other factors.
    """

    def accept(self, visitor, *args, **kwargs):
        """
        Accept a visitor according to the
        U{visitor design pattern<http://en.wikipedia.org/wiki/Visitor_pattern>}.
        """
        return visitor.visit_factor_mixture(self, *args, **kwargs)

    def __init__(self, x, factors, _lambda):
        """
        Set which variables represent x and p in this mixture factor.

        @arg x: The vertex representing the target, i.e. the variable that is a mixture.
        @arg factors: The factors to be mixed.
        @arg _lambda: The vertex representing _lambda, the mixing variable.
        """
        self.x = x
        self.factors = list(factors)
        self._lambda = _lambda

    def neighbours(self):
        "@return: The vertices of this factor's variables"
        from itertools import chain
        result = set()
        for f in self.factors:
            for n in f.neighbours():
                result.add(n)
        result.add(self._lambda)
        return result

    @assert_result_finite
    def calculate_update_for(self, v, var_dists):
        """
        Get an update for the variable whose vertex is v.
        """
        if self.x == v:
            _lambda = var_dists[self._lambda].get_expectation()
            y = numpy.array([f.calculate_update_for(v, var_dists)
                             for f in self.factors])
            return _lambda * y
        if self._lambda == v:
            log_likelihoods = numpy.array(
                [f.log_likelihood(var_dists) for f in self.factors])
            #import IPython; IPython.Debugger.Pdb().set_trace()
            return log_likelihoods
        else:
            # v must be a neighbour of one of the factors...
            _lambda = var_dists[self._lambda].get_expectation()
            for f, p in zip(self.factors, _lambda):
                if v in f.neighbours():
                    # it is a neighbour of this factor
                    return p * f.calculate_update_for(v, var_dists)
        raise RuntimeError(
            'Neighbour, v, not known by factor. Cannot calculate an update for unknown vertex.')

    @assert_result_finite
    def log_likelihood(self, var_dists):
        "@return: The log likelihood of this factor under the variational distribution."
        _lambda = var_dists[self._lambda].get_expectation()
        log_likelihoods = numpy.array(
            [f.log_likelihood(var_dists) for f in self.factors])
        return numpy.dot(_lambda, log_likelihoods)


class FactorDirichlet(Factor):
    """
    A factor in a factor graph that represents a Dirichlet distribution.
    """

    def accept(self, visitor, *args, **kwargs):
        """
        Accept a visitor according to the
        U{visitor design pattern<http://en.wikipedia.org/wiki/Visitor_pattern>}.
        """
        return visitor.visit_factor_dirichlet(self, *args, **kwargs)

    def __init__(self, p, u):
        """
        Set which variables represent p and u in this Dirichlet distribution.

        @arg p: The vertex representing p.
        @arg u: The vertex representing u.
        """
        self.p = p
        self.u = u

    def neighbours(self):
        "@return: The vertices of this factor's variables"
        return [self.p, self.u]

    @assert_result_finite
    def log_likelihood(self, var_dists):
        "@return: The log likelihood of this factor under the variational distribution."
        log_p = var_dists[self.p].get_expectation()
        u, log_gamma_u, log_gamma_U = var_dists[self.u].get_expectation()
        return (
            numpy.dot(u - 1., log_p)
            + log_gamma_U
            - log_gamma_u.sum()
        )

    @assert_result_finite
    def calculate_update_for(self, v, var_dists):
        """
        Get an update for the variable whose vertex is v.
        """
        if self.p == v:
            u, log_gamma_u, log_gamma_U = var_dists[self.u].get_expectation()
            return u - 1.
        if self.u == v:
            log_p = var_dists[self.p].get_expectation()
            return log_p
        raise RuntimeError(
            'Neighbour, v, not known by factor. Cannot calculate an update for unknown vertex.')

    @staticmethod
    def natural_p(p):
        "Convert from canonical to natural form."
        return numpy.log(p)

    @staticmethod
    def natural_u(u):
        "Convert from canonical to natural form."
        return (
            u,
            scipy.special.gammaln(u),
            scipy.special.gammaln(u.sum()),
        )


class GraphLabeller(object):
    """A visitor that labels vertices in the graph."""

    def __init__(self, g, label_edges=True):
        self.g = g
        self.label_edges = label_edges
        self.labels = g.add_vertex_property(name='label', type='string')
        self.edge_labels = g.add_edge_property(name='label', type='string')
        self.shapes = g.add_vertex_property(name='shape', type='string')
        #self.fill_colors = g.add_vertex_property(name='fillcolor', type='string')
        self.styles = g.add_vertex_property(name='style', type='string')

    def visit_factor_gaussian(self, factor, v):
        self.labels[v] = "Gaussian"
        self.shapes[v] = "box"
        if self.label_edges:
            self.edge_labels[self.g.edge(v, factor.x)] = 'x'
            self.edge_labels[self.g.edge(v, factor.mu)] = 'mean'
            self.edge_labels[self.g.edge(v, factor.gamma)] = 'precision'

    def visit_factor_gamma(self, factor, v):
        self.labels[v] = "Gamma"
        self.shapes[v] = "box"
        if self.label_edges:
            self.edge_labels[self.g.edge(v, factor.x)] = 'x'
            self.edge_labels[self.g.edge(v, factor.a)] = 'a'
            self.edge_labels[self.g.edge(v, factor.b)] = 'b'

    def visit_factor_mixture(self, factor, v):
        for f in factor.factors:
            f.accept(self, v)
        self.labels[v] = "Mixture"
        self.shapes[v] = "box"
        if self.label_edges:
            self.edge_labels[self.g.edge(v, factor.x)] = 'x'
            self.edge_labels[self.g.edge(v, factor._lambda)] = 'lambda'

    def visit_var_dist_observed(self, var_dist, v):
        #self.fill_colors[v] = 'gray'
        self.styles[v] = 'filled'

    def visit_var_dist_hidden(self, var_dist, v):
        pass


class VariationalUpdater(object):

    _L_check_tolerance = 1.e-8
    "The tolerance we have when checking that L(Q) always increases inline with the variational updates."

    def __init__(self, model, var_dists):
        self.model = model
        "The model."

        self.var_dists = var_dists
        "The variational distributions."

    def update_vertex(self, v):
        # are we checking that the LL increases?
        if __debug__:
            pre_L = self.L()
        # add up all the incoming updates
        self.var_dists[v].update(
            sum(
                self.model.factors[u].calculate_update_for(v, self.var_dists)
                for u
                in self.model.g.adjacent_vertices(v)
            )
        )
        # are we checking that the LL increases?
        if __debug__:
            post_L = self.L()
            if pre_L > post_L + VariationalUpdater._L_check_tolerance:
                raise RuntimeError(
                    'L(Q) decreased by %f as result of variational update' % (pre_L - post_L))

    def update_all(self):
        for v in self.model.g.vertices:
            if None != self.var_dists[v]:
                self.update_vertex(v)

    @assert_result_finite
    def log_likelihood(self):
        "@return: The log likelihood of the factors w.r.t. the variational distribution."
        return sum(
            self.model.factors[v].log_likelihood(self.var_dists)
            for v
            in self.model.factor_vertices()
        )

    def entropy(self):
        "@return: The entropy of the variational distribution."
        return sum(
            self.var_dists[v].entropy()
            for v
            in self.model.variable_vertices()
        )

    def L(self):
        "@return: L(Q) = entropy(variational dist.) + log likelihood"
        return self.entropy() + self.log_likelihood()


if '__main__' == __name__:
    #
    # Set up model
    #
    g = bgl.Graph()
    variables = g.add_vertex_property(type='object')
    factors = g.add_vertex_property(type='object')
    factor_graph = FactorGraph(g, variables, factors)
    mu1 = factor_graph.add_variable(Variable())
    gamma1 = factor_graph.add_variable(Variable())
    mu2 = factor_graph.add_variable(Variable())
    gamma2 = factor_graph.add_variable(Variable())
    K = 5
    xs = [factor_graph.add_variable(Variable()) for i in range(K)]
    _lambdas = [factor_graph.add_variable(Variable()) for i in range(K)]
    for x, _lambda in zip(xs, _lambdas):
        factor_graph.add_factor(
            FactorMixture(
                x,
                (
                    FactorGaussian(x, mu1, gamma1),
                    FactorGaussian(x, mu2, gamma2)
                ),
                _lambda
            )
        )

    #
    # Set up data and variational distributions
    #
    var_dists = factor_graph.g.add_vertex_property(type='object')
    var_dists[mu1] = VarDistGaussian()
    var_dists[gamma1] = VarDistObserved(FactorGaussian.natural_gamma(1.))
    var_dists[mu2] = VarDistGaussian()
    var_dists[mu2].parameters = numpy.array([-1., -.5])
    var_dists[gamma2] = VarDistObserved(FactorGaussian.natural_gamma(1.))
    var_dists[xs[0]] = VarDistObserved(FactorGaussian.natural_x(-2.1))
    var_dists[xs[1]] = VarDistObserved(FactorGaussian.natural_x(-1.9))
    var_dists[xs[2]] = VarDistObserved(FactorGaussian.natural_x(1.9))
    var_dists[xs[3]] = VarDistObserved(FactorGaussian.natural_x(2.))
    var_dists[xs[4]] = VarDistObserved(FactorGaussian.natural_x(2.1))
    for _lambda in _lambdas:
        var_dists[_lambda] = VarDistDiscrete(K=2)
        var_dists[_lambda].update(numpy.random.uniform(size=2) + 1.)

    #
    # label and draw graph
    #
    labeller = GraphLabeller(factor_graph.g)
    for i, x in enumerate(xs):
        labeller.labels[x] = 'x_%d' % i
    labeller.labels[mu1] = 'mu_1'
    labeller.labels[gamma1] = 'gamma_1'
    labeller.labels[mu2] = 'mu_2'
    labeller.labels[gamma2] = 'gamma_2'
    for i, _lambda in enumerate(_lambdas):
        labeller.labels[_lambda] = 'lambda_%d' % i
    for v in factor_graph.factor_vertices():
        factor_graph.factors[v].accept(labeller, v)
    for v in factor_graph.variable_vertices():
        var_dists[v].accept(labeller, v)
    factor_graph.g.write_graphviz('model.dot')
    os.system(
        'neato -Goverlap=scale -Efontname=arial -Efontsize=8 -Elen=2 -T svg model.dot -o model.svg')

    #
    # Do updates
    #
    variational_updater = VariationalUpdater(factor_graph, var_dists)
    convergence_test = LlConvergenceTest()
    L = variational_updater.L()
    print('log P(D) >= %f' % L)
    convergence_test(L)
    for i in range(40):
        print('************** Update %3d **************' % i)
        variational_updater.update_all()
        L = variational_updater.L()
        print('log P(D) >= %f' % L)
        for i, _lambda in enumerate(_lambdas):
            print('lambda_%i: %s' % (i, var_dists[_lambda].parameters))
        print('mu1   : expected = %f; precision = %f' % (
            FactorGaussian.extract_parameters(var_dists[mu1].parameters)))
        print('mu2   : expected = %f; precision = %f' % (
            FactorGaussian.extract_parameters(var_dists[mu2].parameters)))
        if convergence_test(L):
            break
