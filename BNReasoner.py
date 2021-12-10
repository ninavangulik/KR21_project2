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
################################
### Task 1d
################################

    def summing_out(self, f, B):                
        """ Sum out everything from f except for B 
            
            input: f, a factor
                   B, a variable

            output: f, a factor 
        
        """

        f = f.groupby(B)['p'].sum()
        
        return f

    def multiplying(self, A, B):                
        """ Multiply the factors of A and B 
        
            input: A and B, which are either variables (e.g., 'Winter?') or factors

            output: f, a factor which is a multiplication of A and B

        """

        if type(A) == str and type(B) == str:           # if the input is only a variable
            cpt_A = self.bn.get_cpt(A)
            cpt_B = self.bn.get_cpt(B)
            C = A
        else:                                           # if the input is already a factor
            cpt_A = A
            cpt_B = B
            C = list(cpt_A.keys())[-2]

        f = cpt_A.merge(cpt_B, on=C) 
        f['p'] = f['p_x'] * f['p_y']
        f = f.drop(columns=['p_x', 'p_y'])
        
        return f      

    def get_path(self, Q):   
        """ For all q in Q, get the path to the root. 

            Input: Q, a list of variables, e.g., ['Winter?', 'Slippery Road?']
            Out: master_path, a list of paths, e.g. [['Winter?'], ['Winter?', 'Rain?', 'Slippery Road?']]
       
        """

        G = self.bn.structure
        root = [n for n,d in G.in_degree() if d==0]
        root = ''.join(root)
        master_path_dict = {}
        master_path_list = []
        
        for q in Q:                        
            if q == root:              
                path = [root]           
                master_path_dict[q] = path
                master_path_list.append(path)
            else:
                for path in networkx.all_simple_paths(G, source=root, target=q):
                    master_path_list.append(path)
                    if q not in master_path_dict:
                        master_path_dict[q] = path
                    else:
                        values = master_path_dict.get(q)
                        new_values = [values, path]
                        master_path_dict[q] = new_values
        
        return master_path_list   

    def get_factors(self, master_path):
        """ Get the factors of all the variables in a path

            input: master_path, e.g. [['Winter?'], ['Winter?', 'Rain?', 'Slippery Road?']]
            output: factor_dict, a dictionary that includes the factors of the variables in a path
        
        """
        factor_dict = {}
        
        for path in master_path:            

            path = path[:-1]    
            if len(path) == 1:
                path=[]
                      
            while path:         
                for i in path:  
                    idx = 0
                    f = self.multiplying(path[idx], path[idx+1])   
                    f = self.summing_out(f, path[idx+1])  
                    factor_dict[path[idx+1]] = f
                    path.remove(path[idx])
                    path.remove(path[idx])

        return factor_dict      

    def multiplying2(self, A, B, C):            
        """ Multiply the factors of A and B, including a variable C to merge on.
            
                input: A and B, which are either variables (e.g., 'Winter?') or factors, and C, a variable to merge on

                output: f, a factor which is a multiplication of A and B

        """

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

    def summing_out2(self, f, cols, B, var):    
        """ Sum out B from f.
            
            input: f, a factor
                   cols, the columns to group factor f by
                   B, the variable to be dropped
                   var, the variable the factor f belongs to

            output: f, a factor 
        
        """

        if not var in cols:
            cols.append(var)
        else:
            pass
        f = f.drop(columns=B)
        f = f.groupby(cols, as_index=False)['p'].sum()
        return f

    def multiplying3(self, A, B):                # input: variable/factor1, variable/factor2, variable to merge on
        """ to be used for cross multiplication """


        cpt_A = A
        cpt_B = B

        f = cpt_A.merge(cpt_B, how='cross')
        print(f)

    def marginal_distribution(self, Q, e):
        # Get root of the network
        G = self.bn.structure
        root = [n for n,d in G.in_degree() if d==0]
        root = ''.join(root)
        
        k_factors = {}
        q = None
        f = None

        # Get the paths
        master_path = self.get_path(Q)

        # Iterative over all variables in the query, and append their cpt's to k_factors
        for q in Q:
            if q == root:
                k_factors[q] = self.bn.get_cpt(root)
            else:
                q = q
                f = self.bn.get_cpt(q) # e.g., "Rains?"

                cols = list(f.columns)  # e.g., ["Winter?", "Rains?", "p"]
                cols = cols[:-2]        # e.g., ["Winter?"]
                length = cols           

                if cols[0] == root:     # If the root is in the cpt of q, that means there is a direct connection: we can immediately multiply and sum out
                    root_cpt = self.bn.get_cpt(root)
                    f = self.multiplying(root_cpt, f)
                    f = self.summing_out(f, q)
                    k_factors[q] = f
                
                else:                                                           # If the root is not in the cpt of q:
                    factor_dict = self.get_factors(master_path)                 # get the cpts of the variables in the path (this function excludes the root)

                    for i in range(0, len(length)):                             # Iteratively multiply/sum out
                        key = cols[0]
                        f = self.multiplying2(factor_dict[key], f, key)
                        cols.remove(key)
                        f = self.summing_out2(f, cols, key, q)
                        k_factors[q] = f
            
        if len(k_factors) == 1:                                                 # Solution found
            print(k_factors)
        else:
            while k_factors:                                                    # Otherwise, we still need to multiply what's in k_factors
                first_key = next(iter(k_factors))
                f = k_factors[first_key]
                

                second_key = list(k_factors.keys())[1]
    
                f2 = k_factors[second_key]
                k_factors.pop(first_key, second_key)


################################
### Task 1e
################################

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
