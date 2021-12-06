from itertools import combinations
from typing import Union
import networkx
from BayesNet import BayesNet


class BNReasoner:
    def __init__(self, net: Union[str, BayesNet]):
        """
        :param net: either file path of the bayesian network in BIFXML format or BayesNet object
        """
        if type(net) == str:
            # constructs a BN object
            self.bn = BayesNet()
            # Loads the BN from an BIFXML file
            self.bn.load_from_bifxml(net)
        else:
            self.bn = net

    # TODO: This is where your methods should go

    def is_sequential(self, path):
        _1, _2, _3 = path
        return _2 in self.bn.get_children(_1) and _3 in self.bn.get_children(_2)

    def is_divergent(self, path):
        _1, _2, _3 = path
        return self.bn.get_children(_2) == [_1, _3]

    def is_convergent(self, path):
        _1, _2, _3 = path
        return _2 in self.bn.get_children(_1) and _2 in self.bn.get_children(_3)

    def is_path_closed(self, path, z):
        if self.is_sequential(path):
            return True if z == path[1] else False

        elif self.is_divergent(path):
            return True if z == path[1] else False

        elif self.is_convergent(path):
            return True if z not in path else False

    def d_seperable(self, x, z, y):
        # Create an undirected networkx graph to calculate all possible paths from x to y
        nx = self.bn.structure.to_undirected()
        all_paths = list(networkx.algorithms.simple_paths.all_simple_paths(nx, x, y))

        # Check if each path has a closed sub_path
        window_size = 3
        closed_paths = []
        for path in all_paths:
            for i in range(len(path) - window_size + 1):
                sub_path = path[i: i + window_size]
                is_closed = self.is_path_closed(sub_path, z)
                if is_closed:
                    closed_paths.append(path)
                    break

        if len(closed_paths) == len(all_paths):
            return True
        else:
            return False

    def min_degree_order(self):
        G = self.bn.get_interaction_graph()
        X = self.bn.get_all_variables()

        pi = []
        for i in range(len(X)):
            # sort variables on number of neighbors and append minimum to pi
            dict_of_neighbors = {var: list(networkx.neighbors(G, var)) for var in X}
            sorted_neighbors = sorted([(key, len(value)) for key, value in dict_of_neighbors.items()], key=lambda x: x[1])
            pi.append(sorted_neighbors[0][0])

            # add an edge between every pair of non-adjacent neighbors
            if len(dict_of_neighbors[pi[-1]]) >= 2:
                pairwise_edges = list(combinations(dict_of_neighbors[pi[-1]], 2))
                print(pairwise_edges)
                [G.add_edge(*pair) for pair in pairwise_edges]

            # remove node from graph and variable list
            G.remove_node(pi[-1])
            X.remove(pi[-1])

        return pi

    def min_fill_order(self):
        G = self.bn.get_interaction_graph()
        X = self.bn.get_all_variables()

        pi = []
        for i in range(len(X)):
            dict_of_neighbors = {var: list(networkx.neighbors(G, var)) for var in X}

            edges_to_add = []
            for var in X:
                pairwise_edges = list(combinations(dict_of_neighbors[var], 2))
                n_edges = sum([pair not in G.edges for pair in pairwise_edges])  # maybe add: pair[::-1] not in G_copy.edges
                edges_to_add.append((var, n_edges))

            edges_to_add = sorted(edges_to_add, key=lambda x: x[1])
            print(edges_to_add)
            pi.append(edges_to_add[0][0])

            # add an edge between every pair of non-adjacent neighbors
            if len(dict_of_neighbors[pi[-1]]) >= 2:
                pairwise_edges = list(combinations(dict_of_neighbors[pi[-1]], 2))
                [G.add_edge(*pair) for pair in pairwise_edges]

            # remove node from graph and variable list
            G.remove_node(pi[-1])
            X.remove(pi[-1])

        return pi

    def prune_nodes(self, Q, e):
        """ Pruning all leaf nodes not in Q or e. """

        X = self.bn.get_all_variables()
        leaf_nodes = [x for x in X if len(self.bn.get_children(x)) == 0]

        nodes_to_delete = []
        for node in leaf_nodes:
            # check if node is part of query Q
            if node in Q:
                continue
            # check if node is part of evidence e
            elif any([(node in x) for x in e]):
                continue
            # if both False, than delete leaf node
            else:
                nodes_to_delete.append(node)

        [self.bn.del_var(node) for node in nodes_to_delete]

        return self

    def prune_edges(self, e):
        """ Pruning all outgoing edges from evidence nodes e. """

        # Remove outgoing edges from evidence
        edges_to_remove = {node[0]: self.bn.get_children(node[0]) for node in e}

        for key, value in edges_to_remove.items():
            for edge_node in value:
                self.bn.del_edge((key, edge_node))

        # Update CPT tables
        cpts = self.bn.get_all_cpts()
        for node, cpt in cpts.items():
            if len(cpt.columns) > 2:
                for evidence in e:
                    if evidence[0] not in cpt.columns[-2]:
                        if evidence[0] in cpt.columns:
                            cpt = cpt.loc[lambda d: d[evidence[0]] == evidence[1]]
                self.bn.update_cpt(node, cpt)

        return self

    def prune_network(self, Q, e):
        self.prune_nodes(Q, e)
        self.prune_edges(e)

        return self


if __name__ == "__main__":
    reasoner = BNReasoner(net="./testing/lecture_example.BIFXML")

    Q = ["Wet Grass?"]
    e = [("Winter?", True), ("Rain?", False)]

    reasoner.prune_network(Q, e)
    print(reasoner.bn.get_all_cpts())