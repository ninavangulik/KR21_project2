from itertools import combinations
from os import R_OK, path
from typing import Union
import networkx
from BayesNet import BayesNet
import pandas as pd


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

    def multiplying(self, A, B, C):                # input: variable/factor1, variable/factor2, variable to merge on
        if type(A) == str and type(B) == str:
            cpt_A = self.bn.get_cpt(A)
            cpt_B = self.bn.get_cpt(B)
            C = A
        else:
            cpt_A = A
            cpt_B = B

        f = cpt_A.merge(cpt_B, on=C) 
        f['p'] = f['p_x'] * f['p_y']
        f = f.drop(columns=['p_x', 'p_y'])
        
        return f
    
    def multiplying2(self, A, B, first, second):                # input: variable/factor1, variable/factor2, variable to merge on
        cpt_A = A
        cpt_B = B

        rows = [True, True, False, False]
        rows2 = [True, False, True, False]
        a_values = []
        b_values = []
        
        for i in rows:
            a_values.append(float(cpt_A.loc[cpt_A[first] == i]['p']))
                
        for j in rows2:
            b_values.append(float(cpt_B.loc[cpt_B[second] == j]['p']))
        
        f = pd.DataFrame(columns=[first, second, 'p'])
        f[first] = rows
        f[second] = rows2

        p_values = []

        for num1, num2 in zip(a_values, b_values):
            p_values.append(num1 * num2)

        f['p'] = p_values

    def summing_out(self, f, B):                # keep only B
        f = f.groupby(B)['p'].sum()

        return f

    def summing_out2(self, f, cols, B, var):               # remove B
        if not var in cols:
            cols.append(var)
        else:
            pass
        f = f.drop(columns=B)
        f = f.groupby(cols, as_index=False)['p'].sum()
        return f

    def get_path(self, Q):                  # get predecessors of Q up until the root
        G = self.bn.structure
        root = [n for n,d in G.in_degree() if d==0]
        root = ''.join(root)
        master_path = []
        
        for q in Q:                     #Q = ["Winter?", "Rain?", "Slippery Road?"], q = "Winter?"... 
            if q == root:              
                path = [root]           
                master_path.append(path)
            else:
                for path in networkx.all_simple_paths(G, source=root, target=q):
                    master_path.append(path)
                        
        return master_path

    def get_factor_dict(self, master_path): # [['Winter?'], ['Winter?', 'Rain?'], ['Winter?', 'Rain?', 'Slippery Road?']]  
        factor_dict = {}
        
        for path in master_path:            # e.g. ['Winter?']

            path = path[:-1]    
            if len(path) == 1:
                path=[]
                      
            while path:         
                for i in path:  
                    idx = 0
                    f = self.multiplying(path[idx], path[idx+1], path[idx])   
                    f = self.summing_out(f, path[idx+1])                      
                    factor_dict[path[idx+1]] = f
                    path.remove(path[idx])
                    path.remove(path[idx])

        return factor_dict

    def marginal_distribution(self, Q, e): 
        G = self.bn.structure
        root = [n for n,d in G.in_degree() if d==0]
        root = ''.join(root)
        f = None
        q = None
        master_path = self.get_path(Q)
        master_f = {}

        for q in Q: 
            if q == root:
                master_f[q] = self.bn.get_cpt(root)
                #print(self.bn.get_cpt(root))
            else:
                q = q
                f = self.bn.get_cpt(q) # factor 'winter, rains'

                cols = list(f.columns)
                cols = cols[:-2] # to remove
                length = cols

                if cols[0] == root:
                    root_cpt = self.bn.get_cpt(root)
                    f = self.multiplying(f, root_cpt, root)
                    f = self.summing_out(f, q)
                    master_f[q] = f
                else:
                    factor_dict = self.get_factor_dict(master_path)
                    for i in range(0, len(length)):
                        key = cols[0]
                        f = self.multiplying(f, factor_dict[key], key)
                        cols.remove(key)
                        f = self.summing_out2(f, cols, key, q)
                        master_f[q] = f
        
        if len(master_f) == 1:
            print(master_f)
        else:
            for i in range(0, len(master_f)):
                first_key = next(iter(master_f))
                f = master_f[first_key]
                if 1 < len(master_f):
                    second_key = list(master_f.keys())[1]
                    print("second", second_key)
                    second = master_f[second_key]
                    print(second)
                    master_f.pop(first_key)
            #f = self.multiplying2(f, second, first_key, second_key)
                f = self.multiplying2(f, second, first_key, second_key)
        print(f)

    def maxing_out(self, cpt, key=None):
        if key is None:
            return cpt.loc[lambda d: d["p"] == d["p"].max()]
        else:
            return cpt.groupby(key).max().reset_index()

    def multiply_new(self, cpt_left: pd.DataFrame, cpt_right: pd.DataFrame, key: str) -> pd.DataFrame:
        if type(cpt_left) != pd.DataFrame:
            raise (TypeError(f"{cpt_left} should be of type pd.DataFrame"))
        if type(cpt_right) != pd.DataFrame:
            raise (TypeError(f"{cpt_right} should be of type pd.DataFrame"))

        return (
            cpt_left
                .merge(cpt_right, on=key)
                .assign(p=lambda d: d["p_x"] * d["p_y"])
                .drop(["p_x", "p_y"], axis=1)
        )

    def calculate_MAP(self):
        ...

    def calculate_MPE(self, e):
        """ Calculating the Most Probable Explanations in a Bayesian Network. """
        self.prune_edges(e)

        # Remove rows were evidence is incompatible for tables <= 2
        cpts = self.bn.get_all_cpts()
        for node, cpt in cpts.items():
            if len(cpt.columns) <= 2:
                for evidence in e:
                    if evidence[0] in cpt.columns:
                        cpt = cpt.loc[lambda d: d[evidence[0]] == evidence[1]]
                self.bn.update_cpt(node, cpt)

        Q = self.bn.get_all_variables()
        pi = self.min_degree_order()

        print("Q", Q)
        print("pi", pi)

        S = self.bn.get_all_cpts()

        # cpt_ = self.maxing_out(S["Wet Grass?"], key="Wet Grass?")
        # print(cpt_)

        for var in pi[1:]:
            cpt_with_var = [key for key, value in S.items() if var in value.columns]
            print(cpt_with_var)
            if len(cpt_with_var) > 1:
                foo = self.multiply_new(S[cpt_with_var[0]], S[cpt_with_var[1]], var)
                print(foo)
            # S[var] = self.maxing_out(S[var])
            # print(S[var])
            break


if __name__ == "__main__":
    reasoner = BNReasoner(net="./testing/lecture_example.BIFXML")

    Q = ["Winter?", "Slippery Road?"]
    e = [("Winter?", True), ("Rain?", False)]
    #reasoner.get_path(Q)
    # reasoner.prune_network(Q, e)
    # print(reasoner.bn.get_all_cpts())
    # reasoner.marginal_distribution(Q, e)

    reasoner.calculate_MPE(e)
