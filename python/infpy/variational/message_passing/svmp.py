#
# Copyright John Reid 2008
#

"""
Code for structured variational message passing.
"""

import boost.graph as bgl
import os



def write_graph(g, graph_name):
    g.write_graphviz('%s.dot' % graph_name)
    os.system('neato -Goverlap=scale -Elen=2 -T svg %s.dot -o %s.svg' % (graph_name, graph_name))



def copy_graph_structure(g):
    """
    Copy the graph and create maps between the vertices of the graph and its copy.

    Only copies the structure not the properties

    @return: A 3-tuple (copy of g, map from g to copy, map from copy to g)
    """
    copy = bgl.Graph()
    copy2g = copy.add_vertex_property(type='vertex')
    g2copy = g.add_vertex_property(type='vertex')
    for v in g.vertices: # add the vertices
        u = copy.add_vertex()
        copy2g[u] = v
        g2copy[v] = u
    for e in g.edges: # add the edges
        copy.add_edge(
          g2copy[g.source(e)],
          g2copy[g.target(e)]
        )
    return copy, g2copy, copy2g



def maximum_cardinality_search(g):
    """
    Find the vertex with the maximum number of neighbours. Used as a heuristic to order vertices during
    triangulation.
    """
    return max((g.in_degree(v), v) for v in g.vertices)[1]




def minimum_deficiency_search(g):
    """
    Find the vertex that will incur the least number of edges to be added to the graph during
    triangulation.
    """
    def induced_neighbour_metric(g, v):
        """
        Return |V|**2 - 2*|E| for the subgraph induced on g by v's neighbours. Used as a heuristic to
        choose next vertex for triangulation.
        """
        neighbours = list(g.adjacent_vertices(v))
        result = len(neighbours)**2
        for u1 in neighbours:
            for u2 in neighbours:
                if None != g.edge(u1, u2):
                    result -= 2
        return result
    return min((induced_neighbour_metric(g, v), v) for v in g.vertices)[1]



def edges_to_triangulate(g):
    """
    Triangulate a graph such that is it chordal,
    i.e. no there is no cycle of length 4 or greater without a chord

    @arg g: The graph.
    @return: (yields) A list of pairs of vertices that when connected will result in a triangulated graph.
    """
    copy, g2copy, copy2g = copy_graph_structure(g)
    vertex_visit_order = [v for v in copy.vertices]
    while copy.num_vertices() > 1:
        # choose a v
        # v = maximum_cardinality_search(copy) # not supposed to be as good a heuristic as minimum_deficiency_search
        v = minimum_deficiency_search(copy)
        for u1 in copy.adjacent_vertices(v):
            for u2 in copy.adjacent_vertices(v):
                if u1 != u2:
                    if None == copy.edge(u1, u2): # if no edge
                        # add an edge
                        copy.add_edge(u1, u2)
                        yield (copy2g[u1], copy2g[u2])
        copy.clear_vertex(v)
        copy.remove_vertex(v)


def triangulate(g):
    """
    Triangulate a graph while yielding the edges added.
    """
    for u, v in edges_to_triangulate(g):
        yield g.add_edge(u, v)




def test_triangulation():
    g = bgl.Graph()
    labels = g.add_vertex_property(name='label', type='string')
    styles = g.add_edge_property(name='style', type='string')
    vs = [g.add_vertex() for i in xrange(5)]
    for i, v in enumerate(vs):
        labels[v] = '%d' % i
    g.add_edge(vs[0],vs[1])
    g.add_edge(vs[0],vs[3])
    g.add_edge(vs[1],vs[2])
    g.add_edge(vs[2],vs[3])
    g.add_edge(vs[3],vs[4])
    write_graph(g, triangulation_before)
    for e in triangulate(g):
        styles[e] = 'dashed'
    write_graph(g, triangulation_after)






def enumerate_maximal_cliques(g):
    """
    Yield the maximal cliques in the graph.

    Tomita: U{The worst-case time complexity for generating all maximal cliques<http://tinyurl.com/5zhj5w>}

    U{Overview of maximal clique algorithms<http://tinyurl.com/5lrwwb>} - page 19.

    @return: Yields sets of vertices, each of which forms a maximal clique.
    """
    Q = set()
    def expand(SUBG, CAND):
        if not SUBG:
            yield Q
        else:
            u = max((len(CAND.intersection(g.adjacent_vertices(u))), u) for u in SUBG)[1]
            EXTu = CAND.difference(g.adjacent_vertices(u))
            while EXTu:
                q = EXTu.pop()
                Q.add(q)
                SUBGq = SUBG.intersection(g.adjacent_vertices(q))
                CANDq = CAND.intersection(g.adjacent_vertices(q))
                for C in expand(SUBGq, CANDq):
                    yield C
                CAND.remove(q)
                Q.remove(q)
    for C in expand(set(g.vertices), set(g.vertices)):
        yield C





def construct_junction_tree(cliques):
    "Construct a junction tree from a sequence of cliques."
    # construct a complete graph which has the clique intersections as edge weights
    g = bgl.Graph()
    clusters = g.add_vertex_property(type='object')
    for clique in cliques:
        clusters[g.add_vertex()] = set(clique)
    weights = g.add_edge_property(type='float')
    separators = g.add_edge_property(type='object')
    for v1 in g.vertices:
        for v2 in g.vertices:
            if v1 != v2 and None == g.edge(v1, v2):
                intersection = clusters[v1].intersection(clusters[v2])
                if intersection:
                    e = g.add_edge(v1, v2)
                    separators[e] = intersection
                    weights[e] = 1.0 / len(intersection)
    maximal_tree_edges = bgl.kruskal_minimum_spanning_tree(g, weights)
    # remove edges not in maximal tree
    for e in g.edges:
        if e not in maximal_tree_edges:
            g.remove_edge(e)
    return g, clusters, separators




class Factor(object):
    "A factor in a factor graph."
    def __init__(self, neighbours):
        self.neighbours = list(neighbours)

    def accept(self, visitor, *args, **kwargs):
        """
        Accept a visitor according to the
        U{visitor design pattern<http://en.wikipedia.org/wiki/Visitor_pattern>}.
        """
        return visitor.visit_factor(self, *args, **kwargs)





class Variable(object):
    "A variable in a factor graph."

    def __init__(self, name=""):
        self.name = name

    def accept(self, visitor, *args, **kwargs):
        """
        Accept a visitor according to the
        U{visitor design pattern<http://en.wikipedia.org/wiki/Visitor_pattern>}.
        """
        return visitor.visit_variable(self, *args, **kwargs)





class FactorGraph(object):
    "A factor graph."

    def __init__(self):
        self.g = bgl.Graph()
        self.nodes = self.g.add_vertex_property(type='object')

    def add_variable(self, variable):
        "@return: The vertex for the variable."
        v = self.g.add_vertex()
        self.nodes[v] = variable
        return v

    def add_factor(self, factor):
        "@return: The vertex for the factor."
        v = self.g.add_vertex()
        self.nodes[v] = factor
        for u in factor.neighbours:
            self.g.add_edge(v, u)
        return v

    def variable_graph(self):
        """
        Takes a factor graph and creates a graph that gives the dependencies between the variables.

        @return: A tuple: (new graph, vertex map to variables)
        """
        var_g = bgl.Graph()
        var_map = var_g.add_vertex_property(type='object')
        vertex_map = self.g.add_vertex_property(type='vertex')
        # add the vertices
        for v in self.g.vertices:
            if isinstance(self.nodes[v], Variable):
                u = var_g.add_vertex()
                vertex_map[v] = u
                var_map[u] = self.nodes[v]
        # add the edges
        for v in self.g.vertices:
            if isinstance(self.nodes[v], Variable):
                for factor in self.g.adjacent_vertices(v):
                    for v1 in self.g.adjacent_vertices(factor):
                        if v1 != v:
                            u, u1 = vertex_map[v], vertex_map[v1]
                            if None == var_g.edge(u, u1):
                                var_g.add_edge(u, u1)
        return var_g, var_map




class GraphLabeller(object):
    """A visitor that labels vertices in the graph."""
    def __init__(self, g):
        self.g = g
        self.labels = g.add_vertex_property(name='label', type='string')
        self.edge_labels = g.add_edge_property(name='label', type='string')
        self.shapes = g.add_vertex_property(name='shape', type='string')
        #self.fill_colors = g.add_vertex_property(name='fillcolor', type='string')
        self.styles = g.add_vertex_property(name='style', type='string')
        self.edge_styles = g.add_edge_property(name='style', type='string')

    def visit_factor(self, factor, v):
        self.labels[v] = "Factor"
        self.shapes[v] = "box"

    def visit_variable(self, variable, v):
        self.shapes[v] = "circle"
        self.labels[v] = variable.name





def winn_fig_52_model():
    "A factor graph for Figure 5.2 (a) in Winn's thesis."
    model = FactorGraph()

    # variables first...
    A = model.add_variable(Variable(name="A"))
    B = model.add_variable(Variable(name="B"))
    C = model.add_variable(Variable(name="C"))
    D = model.add_variable(Variable(name="D"))
    E = model.add_variable(Variable(name="E"))
    F = model.add_variable(Variable(name="F"))

    # now the factors...
    model.add_factor(Factor([A, B]))
    model.add_factor(Factor([A, C]))
    model.add_factor(Factor([B, D]))
    model.add_factor(Factor([C, E]))
    model.add_factor(Factor([D, F, E]))

    return model



def winn_fig_53_model():
    "A factor graph for Figure 5.3 (a) in Winn's thesis."
    model = FactorGraph()

    # variables first...
    A = model.add_variable(Variable(name="A"))
    B = model.add_variable(Variable(name="B"))
    C = model.add_variable(Variable(name="C"))
    D = model.add_variable(Variable(name="D"))
    E = model.add_variable(Variable(name="E"))
    F = model.add_variable(Variable(name="F"))
    G = model.add_variable(Variable(name="G"))

    # now the factors...
    model.add_factor(Factor([A, B, C]))
    model.add_factor(Factor([C, D]))
    model.add_factor(Factor([D, E]))
    model.add_factor(Factor([E, F]))
    model.add_factor(Factor([E, G]))

    return model



def gatsby_lect5_model():
    "A factor graph for the factor graph in the handout for lecture 5 of the Gatsby course."
    model = FactorGraph()

    # variables first...
    A = model.add_variable(Variable(name="A"))
    B = model.add_variable(Variable(name="B"))
    C = model.add_variable(Variable(name="C"))
    D = model.add_variable(Variable(name="D"))
    E = model.add_variable(Variable(name="E"))

    # now the factors...
    model.add_factor(Factor([A, B]))
    model.add_factor(Factor([B, C]))
    model.add_factor(Factor([C, D, E]))
    model.add_factor(Factor([A, E]))
    model.add_factor(Factor([C]))
    model.add_factor(Factor([D]))

    return model



def asia_chest_clinic_model():
    "Returns a factor graph for the asia chest clinic."
    model = FactorGraph()

    # variables first...
    visit_asia = model.add_variable(Variable(name="visit asia"))
    tuberculosis = model.add_variable(Variable(name="tuberculosis"))
    x_ray = model.add_variable(Variable(name="X-ray"))
    smoking = model.add_variable(Variable(name="smoking"))
    lung_cancer = model.add_variable(Variable(name="lung cancer"))
    tub_or_cancer = model.add_variable(Variable(name="tub. or cancer"))
    bronchitis = model.add_variable(Variable(name="bronchitis"))
    dyspnoea = model.add_variable(Variable(name="dyspnoea"))

    # now the factors...
    model.add_factor(Factor([visit_asia, tuberculosis]))
    model.add_factor(Factor([smoking, lung_cancer]))
    model.add_factor(Factor([x_ray, lung_cancer, tuberculosis]))
    model.add_factor(Factor([tub_or_cancer, lung_cancer, tuberculosis]))
    model.add_factor(Factor([dyspnoea, tub_or_cancer, bronchitis]))

    return model


if '__main__' == __name__:
    #test_triangulation()

    #model = asia_chest_clinic_model()
    #model = winn_fig_52_model()
    model = gatsby_lect5_model()

    #
    # label and draw graph
    #
    labeller = GraphLabeller(model.g)
    for v in model.g.vertices:
        model.nodes[v].accept(labeller, v)
    write_graph(model.g, "svmp_model")



    #
    # Drop the factors and get a graph just of the variables
    #
    var_g, var_map = model.variable_graph()
    labeller = GraphLabeller(var_g)
    for v in var_g.vertices:
        var_map[v].accept(labeller, v)
    write_graph(var_g, "variable_graph")

    #
    # Triangulate the graph
    #
    for ev in triangulate(var_g):
        labeller.edge_styles[e] = 'dashed'
    write_graph(var_g, "triangulated_graph")


    #
    # Find the maximal cliques
    #
    for clique in enumerate_maximal_cliques(var_g):
        print ", ".join(labeller.labels[v] for v in clique)


    #
    # Construct a junction tree
    #
    junction_tree, clusters, separators = construct_junction_tree(enumerate_maximal_cliques(var_g))
    cluster_labels = junction_tree.add_vertex_property(name='label', type='string')
    separator_labels = junction_tree.add_edge_property(name='label', type='string')
    for v in junction_tree.vertices:
        cluster_labels[v] = ":".join(labeller.labels[u] for u in clusters[v])
    for e in junction_tree.edges:
        separator_labels[e] = ":".join(labeller.labels[u] for u in separators[e])
    write_graph(junction_tree, "junction_tree")
