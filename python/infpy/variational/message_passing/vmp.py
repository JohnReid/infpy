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


class Factor(object):
    "A factor in a factor graph."

    @staticmethod
    def accept(visitor, *args, **kwargs):
        """
        Accept a visitor according to the visitor design pattern.

        http://en.wikipedia.org/wiki/Visitor_pattern
        """
        return visitor.visit_factor(*args, **kwargs)


class ConditionalDistribution(Factor):
    "A conditional distribution."
    pass

    def set_variables(self, variables):
        "Set the variables of a conditional distribution."
        self.target = variables[0]
        "The target of the conditional distribution"
        self.set_conditioned(variables[1:])


class GaussianFactor(ConditionalDistribution):
    "A Gaussian (or normal) conditional distribution."

    @staticmethod
    def get_label(variable_names):
        "@return: A label suitable for display"
        return "%s = N(%s, %s)" % tuple(variable_names)

    @staticmethod
    def convert_x(x):
        "Convert to vector form."
        return numpy.array((x, x**2))

    @staticmethod
    def convert_mu(mu):
        "Convert to vector form."
        return numpy.array((mu, mu**2))

    @staticmethod
    def convert_gamma(gamma):
        "Convert to vector form."
        return numpy.array((gamma, numpy.log(gamma)))

    @staticmethod
    def calculate_update(messages, index):
        "Calculate the update that should be sent to connected variable with the given index."
        if 3 != len(messages):
            raise ValueError('Expecting 3 messages')
        x_msg, mu_msg, gamma_msg = messages
        if 0 == index:  # update for x
            gamma = gamma_msg[0]
            mu = mu_msg[0]
            return numpy.array((gamma * mu, -.5 * gamma))
        if 1 == index:  # update for mu
            gamma = gamma_msg[0]
            x = x_msg[0]
            return numpy.array((gamma * x, -.5 * gamma))
        if 2 == index:  # update for gamma
            mu, mu2 = mu_msg
            x, x2 = x_msg
            return numpy.array((mu * x - .5 * (x2 + mu2), .5))
        raise ValueError('Bad index given.')


class GammaFactor(ConditionalDistribution):
    "A Gamma conditional distribution."

    @staticmethod
    def get_label(variable_names):
        "@return: A label suitable for display"
        return "%s = Gamma(%s, %s)" % tuple(variable_names)

    @staticmethod
    def convert_x(x):
        "Convert to vector form."
        return numpy.array((x, numpy.log(x)))

    @staticmethod
    def convert_a(a):
        "Convert to vector form."
        return numpy.array((a, scipy.special.gammaln(a)))

    @staticmethod
    def convert_b(b):
        "Convert to vector form."
        return numpy.array((b, numpy.log(b)))

    @staticmethod
    def calculate_update(messages, index):
        "Calculate the update that should be sent to connected variable with the given index."
        if 3 != len(messages):
            raise ValueError('Expecting 3 messages')
        x_msg, a_msg, b_msg = messages
        if 0 == index:  # update for x
            a = a_msg[0]
            b = b_msg[0]
            return numpy.array((-b, a - 1.))
        if 1 == index:  # update for a
            log_x = x_msg[1]
            log_b = b_msg[1]
            return numpy.array((log_x + log_b, -1.))
        if 2 == index:  # update for b
            x = x_msg[0]
            a = a_msg[0]
            return numpy.array((-x, a))
        raise ValueError('Bad index given.')


class Variable(object):
    """
    A random variable (or perhaps several similar variables if a member of one or more plates).
    """

    def __init__(self, dims, plates=None):
        """
        Initialise this random variable

        @arg dims: he dimensions of the natural statistic vector.
        @arg plates: The plates this variable is indexed by (a sequence of plate names).
        """

        self.dims = dims
        "The dimensions of the natural statistic vector."

        self.plates = None != plates and list(plates) or []
        "A sequence of the names of the plates this variable is a member of."

    def accept(self, visitor, *args, **kwargs):
        """
        Accept a visitor according to the visitor design pattern.

        http://en.wikipedia.org/wiki/Visitor_pattern
        """
        #print args
        return visitor.visit_variable(*args, **kwargs)


class FactorGraph(object):
    """
    Represents a probabilistic model as a factor graph.

    We build a bipartite graph of variables and the distributions that link them. Plates
    are not explicitly represented in the graph.
    """

    def __init__(self):
        self.g = bgl.Graph()
        "The graphical representation of our model."

        self.names = self.g.add_vertex_property(name='label', type='string')
        "Maps vertices to variable names."

        self.data = self.g.add_vertex_property(type='object')
        "Maps vertices to variables/factors."

        self.factor_neighbours = self.g.add_vertex_property(type='object')
        """
        Maps factor vertices to lists of neighbouring vertices. The lists are in the order
        expected by the factor.
        """

        self.shapes = self.g.add_vertex_property(name='shape', type='string')
        "Maps vertices to shapes (for dot language)."

        self.plates = dict()
        "Maps plate names to plate sizes."

        self.variable_vertices = dict()
        "Maps variable names to vertices."

    def _add_plate_if_missing(self, name):
        "Add the named plate to the model if it is not already in it."
        if name not in self.plates:
            self.plates[name] = None

    def set_plate_size(self, name, size):
        "Set the size of the named plate."
        if name not in self.plates:
            raise ValueError('Model does not have plate: %s' % name)
        self.plates[name] = size

    def get_variable_shape(self, name):
        "@return: the shape of the given variable (including expansion via plates)."
        from itertools import chain
        variable = self.variable(name)
        return tuple(
            chain(
                [variable.dims],
                (self.plates[p] for p in variable.plates)
            )
        )

    def add_variable(self, name, variable):
        "Add the named variable to the factor graph."
        # add the vertex to the graph and update the property maps
        v = self.g.add_vertex()
        self.variable_vertices[name] = v
        self.names[v] = name
        assert not isinstance(variable, dict)
        self.data[v] = variable
        # add the necessary plates
        for plate in variable.plates:
            self._add_plate_if_missing(plate)

    def add_factor(self, factor, variable_names):
        """
        Add a factor to the model.

        @arg factor: The factor.
        @arg variable_names: The variables in the factor. These must be in the correct
        order as the factor defines, i.e. mean, precision for a Gaussian.
        """
        factor_v = self.g.add_vertex()
        self.data[factor_v] = factor
        self.shapes[factor_v] = 'box'
        self.names[factor_v] = factor.get_label(variable_names)
        self.factor_neighbours[factor_v] = [
            self.variable_vertices[name] for name in variable_names]
        for v in self.factor_neighbours[factor_v]:
            self.g.add_edge(factor_v, v)

    def _vertex(self, name):
        "@return: the vertex with the given name"
        return self.variable_vertices[name]

    def _are_plates_specified(self):
        "@return: True if and only if the size of all the model's plates are specified."
        return None not in self.plates.values()

    def check_plates_specified(self):
        "Raises exception if not all plates in the model have their sizes specified."
        if not self._are_plates_specified():
            raise ValueError(
                'Plates in model are not specified.\n%s' % str(self.plates))

    def variable_names(self):
        "@return: A sequence of the variable names in the model."
        return self.variable_vertices.keys()

    def variable(self, name):
        "@return: The variable with the given name."
        return self.data[self._vertex(name)]

    def _is_variable(self, v):
        "Is this vertex a factor or a variable?"
        return isinstance(self.data[v], Variable)


def create_factorised_variational_distribution(model, hidden_variables):
    """
    Initialise this factorised variational distribution.

    @arg model: The model that contains the hidden variables.
    @arg hidden_variables: A sequence of names of the hidden variables.
    """
    return dict(
        (hidden, numpy.zeros(model.get_variable_shape(hidden)))
        for hidden
        in hidden_variables
    )


class LocalMessagePasser(object):
    "Implements a message passing algorithm using only local information."

    def __init__(self, g, v, msg_passer_map, name_map=None):
        """
        Initialise the local message passer.

        @arg g: The graph.
        @arg v: The vertex we are passing messages for/through.
        @arg msg_passer_map: A map from graph vertices to LocalMessagePasser objects.
        @arg name_map: A map from vertices to names (for debugging output).
        """

        self.g = g
        "The graph."

        self.v = v
        "The vertex we are passing messages for/through."

        self.msg_passer_map = msg_passer_map
        "A map from graph vertices to LocalMessagePasser objects."

        self.name_map = name_map
        "A map from vertices to names (for debugging output)."

        self.set_adjacent(self.g.adjacent_vertices(self.v))

    def set_adjacent(self, adjacent):
        self.adjacent = list(adjacent)
        "A list of adjacent vertices."

        self.indices = self.g.add_vertex_property(type='integer')
        "A map from vertices to indices into self.adjacent."
        for i, v in enumerate(self.adjacent):
            self.indices[v] = i

    def prepare(self):
        "Reset the message passer ready for another round of message passing."
        self.rcvd = [None] * len(self.adjacent)
        "Keep track of which vertices we have received messages from."

        self.sent = [None] * len(self.adjacent)
        "Keep track of which vertices we have sent messages to."

    def clean_up(self):
        "Called after a round of message passing has completed."
        del self.rcvd
        del self.sent

    def on_receive(self, src, msg):
        "Override to implement event handling when messages are received."
        pass

    def receive_message(self, src, msg):
        """
        Receive a message from src.

        @arg src: The vertex sending the message.
        @arg msg: The message.
        """
        #print 'Received message: %s: %s -> %s' % (msg, self._get_name(src), self._get_name(self.v))
        self.rcvd[self.indices[src]] = msg

        self.on_receive(src, msg)

        # check if the new message enables us to send any messages
        self.send_messages()

    def send_messages(self):
        "Send any messages we are able to calculate."
        num_not_rcvd = self.rcvd.count(None)
        if num_not_rcvd < 2:  # can only send message if received all but one or less
            for v in self.adjacent:
                # if we have not sent this vertex a message and we have received all messages or just not from this one.
                index = self.indices[v]
                if None == self.sent[index] and (0 == num_not_rcvd or None == self.rcvd[index]):
                    msg = self.create_message(v)
                    if None == msg:
                        raise RuntimeError(
                            'Messages cannot be None. Please override create_message().')
                    self.sent[index] = msg  # remember we have sent it
                    print 'Sending  message: %s: %s -> %s' % (
                        msg, self._get_name(self.v), self._get_name(v))
                    self.msg_passer_map[v].receive_message(
                        self.v, msg)  # send it

    def create_message(self, dst):
        "Called to create a message to send to given destination."
        return None

    def check_all_messages_received_and_sent(self):
        "Raises an error if not all messages have been received and sent."
        if self.rcvd.count(None) != 0:
            raise RuntimeError(
                "Did not receive messages from every neighbour.")
        if self.sent.count(None) != 0:
            raise RuntimeError("Did not send messages to every neighbour.")

    def _get_name(self, v):
        if None != self.name_map:
            return self.name_map[v]
        else:
            return '<no name>'


def create_local_message_passers(g, message_passer_creator):
    """
    Prepare a graph for local message passing.

    @arg g: The graph.
    @arg message_passer_creator: A function(g, v, msg_passer_map) that creates local message passers.

    @return: A map from vertices to local message passers.
    """
    msg_passer_map = g.add_vertex_property(type='object')
    for v in g.vertices:
        msg_passer_map[v] = message_passer_creator(g, v, msg_passer_map)
    return msg_passer_map


def pass_local_messages(g, msg_passer_map):
    "Pass messages locally over a graph."
    for v in g.vertices:
        msg_passer_map[v].prepare()
    for v in g.vertices:
        msg_passer_map[v].send_messages()
    for v in g.vertices:
        msg_passer_map[v].check_all_messages_received_and_sent()
    for v in g.vertices:
        msg_passer_map[v].clean_up()


class ObservedVertexMessagePasser(LocalMessagePasser):
    "Produces/consumes messages for an observed vertex."

    def __init__(self, g, v, msg_passer_map, observed, name_map=None):
        LocalMessagePasser.__init__(self, g, v, msg_passer_map, name_map)
        self.observed = observed
        "The observed value."

    def create_message(self, dst):
        return self.observed


class HiddenVertexMessagePasser(LocalMessagePasser):
    "Produces/consumes messages for a hidden vertex."

    def __init__(self, g, v, msg_passer_map, distribution, name_map=None):
        LocalMessagePasser.__init__(self, g, v, msg_passer_map, name_map)
        self.distribution = distribution
        "The variational distribution for this vertex."

    def prepare(self):
        LocalMessagePasser.prepare(self)
        self.msg_sum = numpy.zeros_like(self.distribution)
        "Accumulation of messages received in this message passing round."

    def create_message(self, dst):
        return self.distribution

    def on_receive(self, src, msg):
        self.msg_sum += msg

    def clean_up(self):
        self.distribution[:] = self.msg_sum
        del self.msg_sum
        LocalMessagePasser.clean_up(self)
        print 'New variational distribution: %s: %s' % (
            self._get_name(self.v), self.distribution)


class FactorVertexMessagePasser(LocalMessagePasser):
    "Produces/consumes messages for a factor vertex."

    def __init__(self, g, v, msg_passer_map, factor, neighbours, name_map=None):
        LocalMessagePasser.__init__(self, g, v, msg_passer_map, name_map)
        self.factor = factor
        "The factor."
        self.set_adjacent(neighbours)

    def create_message(self, dst):
        return self.factor.calculate_update(self.rcvd, self.indices[dst])


class VariationalUpdater(object):
    "Performs variational updates."

    def __init__(self, model, observed_values):
        """
        Initialise this variational message passer.

        @arg model: The model we update.
        @arg observed_values: A dict mapping variable names to observed values.
        """
        model.check_plates_specified()

        self.model = model
        "The model."

        self.observed_values = observed_values
        "The observed values in the model."

        self.var_dist = create_factorised_variational_distribution(
            model,
            [
                variable
                for variable
                in self.model.variable_names()
                if variable not in data
            ]
        )
        "Dict mapping hidden variable names to their factorised variational distributions."

        self.msg_passer_map = create_local_message_passers(
            self.model.g, self._create_msg_handler_for)

    def visit_factor(self, *args, **kwargs):
        #print 'Visiting factor'
        g, v, msg_passer_map, data = args
        return FactorVertexMessagePasser(
            g,
            v,
            msg_passer_map,
            data,
            self.model.factor_neighbours[v],
            self.model.names
        )

    def visit_variable(self, *args, **kwargs):
        #print 'Visiting variable'
        #print args
        g, v, msg_passer_map, data = args
        name = self.model.names[v]
        if name in self.observed_values:
            return ObservedVertexMessagePasser(
                g,
                v,
                msg_passer_map,
                self.observed_values[name],
                self.model.names
            )
        else:
            return HiddenVertexMessagePasser(
                g,
                v,
                msg_passer_map,
                self.var_dist[name],
                self.model.names
            )
        print args
        print kwargs

    def _create_msg_handler_for(self, g, v, msg_passer_map):
        data = self.model.data[v]
        return data.accept(self, g, v, msg_passer_map, data)

    def update(self):
        pass_local_messages(self.model.g, self.msg_passer_map)


if '__main__' == __name__:
    model = FactorGraph()
    model.add_variable("m", Variable(dims=2))
    model.add_variable("beta", Variable(dims=2))
    model.add_variable("mu", Variable(dims=2))
    model.add_factor(GaussianFactor, ["mu", "m", "beta"])
    model.add_variable("a", Variable(dims=2))
    model.add_variable("b", Variable(dims=2))
    model.add_variable("gamma", Variable(dims=2))
    model.add_factor(GammaFactor, ["gamma", "a", "b"])
    model.add_variable("x", Variable(dims=2, plates=['N']))
    model.add_factor(GaussianFactor, ["x", "mu", "gamma"])
    N = 1
    model.set_plate_size('N', N)
    model.g.write_graphviz('model.dot')
    os.system('dot -T svg model.dot -o model.svg')

    data = {
        'm': GaussianFactor.convert_mu(0.),
        'beta': GaussianFactor.convert_gamma(1.),
        'a': GammaFactor.convert_a(1.),
        'b': GammaFactor.convert_b(1.),
        'x': GaussianFactor.convert_x(R.normal(1.6, .1, (N,))).sum(axis=1),
    }
    updater = VariationalUpdater(model, data)
    updater.update()
