from collections import Counter
from itertools import combinations
from typing import Union, List

from os import R_OK, path
from loguru import logger
import networkx as nx
import pandas as pd

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

    ################################
    ### Task 1a
    ################################

    def _is_sequential(self, path):
        _1, _2, _3 = path
        return _2 in self.bn.get_children(_1) and _3 in self.bn.get_children(_2)

    def _is_divergent(self, path):
        _1, _2, _3 = path
        return self.bn.get_children(_2) == [_1, _3]

    def _is_convergent(self, path):
        _1, _2, _3 = path
        return _2 in self.bn.get_children(_1) and _2 in self.bn.get_children(_3)

    def _is_path_closed(self, path, z):
        if self._is_sequential(path):
            return True if z == path[1] else False

        elif self._is_divergent(path):
            return True if z == path[1] else False

        elif self._is_convergent(path):
            return True if z not in path else False

    def d_seperable(self, x, z, y):
        """ Check if combination is d-seperable. """
        # Create an undirected networkx graph to calculate all possible paths from x to y
        network = self.bn.structure.to_undirected()
        all_paths = list(nx.algorithms.simple_paths.all_simple_paths(network, x, y))

        # Check if each path has a closed sub_path
        window_size = 3
        closed_paths = []
        for path in all_paths:
            for i in range(len(path) - window_size + 1):
                sub_path = path[i: i + window_size]
                is_closed = self._is_path_closed(sub_path, z)
                if is_closed:
                    closed_paths.append(path)
                    break

        return True if len(closed_paths) == len(all_paths) else False

    ################################
    ### Task 1b
    ################################

    def min_degree_order(self):
        """ Min degree ordering. """
        G = self.bn.get_interaction_graph()
        X = self.bn.get_all_variables()

        pi = []
        for i in range(len(X)):
            # sort variables on number of neighbors and append minimum to pi
            dict_of_neighbors = {var: list(nx.neighbors(G, var)) for var in X}
            sorted_neighbors = sorted([(key, len(value)) for key, value in dict_of_neighbors.items()],
                                      key=lambda x: x[1])
            pi.append(sorted_neighbors[0][0])

            # add an edge between every pair of non-adjacent neighbors
            if len(dict_of_neighbors[pi[-1]]) >= 2:
                pairwise_edges = list(combinations(dict_of_neighbors[pi[-1]], 2))
                [G.add_edge(*pair) for pair in pairwise_edges]

            # remove node from graph and variable list
            G.remove_node(pi[-1])
            X.remove(pi[-1])

        return pi

    def min_fill_order(self):
        """ Min fill ordering. """
        G = self.bn.get_interaction_graph()
        X = self.bn.get_all_variables()

        pi = []
        for i in range(len(X)):
            dict_of_neighbors = {var: list(nx.neighbors(G, var)) for var in X}

            edges_to_add = []
            for var in X:
                pairwise_edges = list(combinations(dict_of_neighbors[var], 2))
                n_edges = sum(
                    [pair not in G.edges for pair in pairwise_edges])  # maybe add: pair[::-1] not in G_copy.edges
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

    ################################
    ### Task 1c
    ################################

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
        # edges_to_remove = {}
        # for node, value in e.iterrows():
        #     edges_to_remove[node] = self.bn.get_children(node)
        edges_to_remove = {node: self.bn.get_children(node) for node, value in e.items()}

        for key, value in edges_to_remove.items():
            for edge_node in value:
                self.bn.del_edge((key, edge_node))

        # Update CPT tables
        cpts = self.bn.get_all_cpts()
        for node, cpt in cpts.items():
            # if len(cpt.columns) > 2:  # TODO: double check if this is necessary
            for evidence, value in e.items():
                # if evidence not in cpt.columns[-2]:  # TODO: double check if this is necessary
                if evidence in cpt.columns:
                    cpt = cpt.loc[lambda d: d[evidence] == value]
            self.bn.update_cpt(node, cpt)

        return self

    def prune_network(self, Q, e):
        self.prune_nodes(Q, e)
        self.prune_edges(e)

        return self

    ################################
    ### Task 1d
    ################################

    def summing_out(self, cpt: pd.DataFrame, key: str) -> pd.DataFrame:
        """ Summing out or marginalizing a cpt for a given key. """
        if type(cpt) != pd.DataFrame:
            raise (TypeError(f"{cpt} should be of type pd.DataFrame"))

        cols_to_group = list(cpt.columns)
        cols_to_group.remove(key)
        cols_to_group.remove("p")
        return (
            cpt
            .drop(key, axis=1)
            .groupby(cols_to_group)
            .sum()
            .reset_index()
        )

    @staticmethod
    def _find_column_intersection(cpt_left: pd.DataFrame, cpt_right: pd.DataFrame) -> list:
        """ Find intersection of column names between to cpt dataframes. """
        left = set(list(cpt_left.columns)[:-1])
        right = set(list(cpt_right.columns)[:-1])

        return list(left.intersection(right))

    def multiplying_factors(self, cpt_left: pd.DataFrame, cpt_right: pd.DataFrame, key=None) -> pd.DataFrame:
        if type(cpt_left) != pd.DataFrame:
            raise (TypeError(f"{cpt_left} should be of type pd.DataFrame"))
        if type(cpt_right) != pd.DataFrame:
            raise (TypeError(f"{cpt_right} should be of type pd.DataFrame"))

        if key is None:
            key = self._find_column_intersection(cpt_left, cpt_right)

        return (
            cpt_left
            .merge(cpt_right, on=key)
            .assign(p=lambda d: d["p_x"] * d["p_y"])
            .drop(["p_x", "p_y"], axis=1)
        )

    def _transform_evidence_into_series(self, e) -> pd.Series:
        """ Transforming evidence e into a pandas series. """
        bools = []
        index = []
        for evidence in e:
            idx, boolean = evidence
            bools.append(boolean)
            index.append(idx)

        return pd.Series(tuple(bools), index=index)

    def _update_cpts_with_evidence(self, S, e):
        """ Update all CPTs according to evidence e. """
        logger.info(f"Update all CPTs according to evidence e")
        for key, value in S.items():
            S[key] = self.bn.get_compatible_instantiations_table(e, value)
        return S

    def marginal_distribution(self, Q: Union[List, str], e=None):
        """ Marginal distribution for a query Q and possible evidence e. """
        logger.info(f"Calculating marginal distribution for query Q: {Q}")

        # input validation: turn Q into a list
        if type(Q) == str:
            Q = [Q]

        S = self.bn.get_all_cpts()
        pi = self.bn.get_all_variables()
        [pi.remove(q) for q in Q]
        # print("pi", pi)

        if e is not None:
            e = self._transform_evidence_into_series(e)
            S = self._update_cpts_with_evidence(S, e)

        # Edge case 1: len(Q) == 1 and Q has no parents: return cpt of Q
        if len(Q) == 1 and nx.algorithms.dag.ancestors(self.bn.structure, Q[0]) == set():
            return S[Q[0]].sort_values(Q[0], ascending=False).reset_index(drop=True)

        # Edge case 2: Q == X: return all cpts multiplied with each other
        if Counter(Q) == Counter(list(S.keys())):
            print("Q == X, so just multiplying all and no summing out")
            cpt_res = S[Q[0]]
            for i in range(1, len(Q)):
                cpt_res = self.multiplying_factors(cpt_res, S[Q[i]])
            return cpt_res.sort_values(Q, ascending=False).reset_index(drop=True)

        cpt_res = None
        for var in pi:
            var_map = {key: list(value.columns)[:-1] for key, value in S.items()}  # [:-1] to remove "p" from the columns list
            var_in_cpts = [key for key, value in var_map.items() if var in value]
            # print("var", var)
            # print("var_in_cpts", var_in_cpts)

            # initialize cpt_res to first cpt
            if cpt_res is None:
                var_in_cpts.remove(var)
                cpt_res = S[var]
                S.pop(var, None)  # remove cpt to prevent that it gets multiplied more than once

            # multiply step
            for mul_var in var_in_cpts:
                # print("mul", mul_var)
                cpt_res = self.multiplying_factors(cpt_res, S[mul_var])
                S.pop(mul_var, None)  # remove cpt to prevent that it gets multiplied more than once

            # summing-out step
            # print("sum", var)
            cpt_res = self.summing_out(cpt_res, key=var)

        # normalize probability values in the case of evidence
        if e is not None:
            cpt_res = cpt_res.assign(p=lambda d: d["p"] / d["p"].sum())

        # sorting for nicer representation
        cpt_res = cpt_res.sort_values(Q, ascending=False).reset_index(drop=True)

        return cpt_res

    ################################
    ### Task 1e
    ################################

    def maxing_out(self, cpt, key=None):
        if key is None:
            return cpt.loc[lambda d: d["p"] == d["p"].max()]
        else:
            return cpt.groupby(key).max().reset_index()

    def maxing_out2(self, cpt: pd.DataFrame, key: str) -> pd.DataFrame:
        """ Summing out or marginalizing a cpt for a given key. """
        if type(cpt) != pd.DataFrame:
            raise (TypeError(f"{cpt} should be of type pd.DataFrame"))

        cols_to_group = list(cpt.columns)
        cols_to_group.remove(key)
        cols_to_group.remove("p")
        if cpt.shape[1] == 2:
            return cpt.loc[lambda d: d["p"] == d["p"].max()]
        else:
            return (
                cpt
                .drop(key, axis=1)
                .groupby(cols_to_group)
                .max()
                .reset_index()
            )

    def calculate_MAP(self, M, e):
        """ Calculating the Maximum A Posteriori in a Bayesian Network. """
        logger.info(f"Calculating MAP for M: {M} and e: {e}")

        if type(e) != pd.Series:
            e = self._transform_evidence_into_series(e)
        self.prune_edges(e)

        S = self.bn.get_all_cpts()
        # pi = self.bn.get_all_variables()
        pi = ["O", "Y", "X", "I", "J"]  # TODO: REMOVE and make general

        # TODO: not sure if this is still necessary for MAP
        if e is not None:
            S = self._update_cpts_with_evidence(S, e)

        cpt_res = None
        for var in pi:
            var_map = {key: list(value.columns)[:-1] for key, value in S.items()}  # [:-1] to remove "p" from the columns list
            var_in_cpts = [key for key, value in var_map.items() if var in value]

            # initialize cpt_res to first cpt
            if cpt_res is None:
                var_in_cpts.remove(var)
                cpt_res = S[var]
                S.pop(var, None)  # remove cpt to prevent that it gets multiplied more than once

            # multiply step
            for mul_var in var_in_cpts:
                cpt_res = self.multiplying_factors(cpt_res, S[mul_var])
                S.pop(mul_var, None)  # remove cpt to prevent that it gets multiplied more than once

            # summing-out or maxing-out step
            if var in M:
                print(f"maxing-out for {var}")
                cpt_res = self.maxing_out2(cpt_res, key=var)  # TODO: check if we need `key=var` here
            else:
                print(f"summing-out for {var}")
                cpt_res = self.summing_out(cpt_res, key=var)

        return cpt_res

    def calculate_MPE(self, e):
        """ Calculating the Most Probable Explanations in a Bayesian Network. """
        logger.info(f"Calculating MPE for evidence e: {e}")

        if type(e) != pd.Series:
            e = self._transform_evidence_into_series(e)
        self.prune_edges(e)

        S = self.bn.get_all_cpts()
        pi = self.bn.get_all_variables()

        # TODO: not sure if this is still necessary for MPE
        if e is not None:
            S = self._update_cpts_with_evidence(S, e)

        cpt_res = None
        for var in pi:
            var_map = {key: list(value.columns)[:-1] for key, value in S.items()}  # [:-1] to remove "p" from the columns list
            var_in_cpts = [key for key, value in var_map.items() if var in value]

            # initialize cpt_res to first cpt
            if cpt_res is None:
                var_in_cpts.remove(var)
                cpt_res = S[var]
                S.pop(var, None)  # remove cpt to prevent that it gets multiplied more than once

            # multiply step
            for mul_var in var_in_cpts:
                cpt_res = self.multiplying_factors(cpt_res, S[mul_var])
                S.pop(mul_var, None)  # remove cpt to prevent that it gets multiplied more than once

            # maxing-out step
            cpt_res = self.maxing_out(cpt_res)  # TODO: check if we need `key=var` here or maxing_out2

        # sorting for nicer representation
        cpt_res = cpt_res.sort_values(pi, ascending=False).reset_index(drop=True)

        return cpt_res


if __name__ == "__main__":
    reasoner = BNReasoner(net="./testing/lecture_example2.BIFXML")

    # Q = ["Wet Grass?", "Slippery Road?"]
    # e = pd.Series((True, False), index=["Winter?", "Sprinkler?"])
    # res = reasoner.marginal_distribution(Q, e)

    # Q = ["O", "Y", "X", "I", "J"]
    # e = [("O", True)]
    # res = reasoner.marginal_distribution(Q, e)
    # print(res)

    # M = ["I", "J"]
    # e = [("O", True)]
    # res = reasoner.calculate_MAP(M, e)
    # print(res)

    e = [("J", True), ("O", False)]
    res = reasoner.calculate_MPE(e)
    print(res)

