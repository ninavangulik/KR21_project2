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


if __name__ == "__main__":
    reasoner = BNReasoner(net="./testing/lecture_example.BIFXML")
    print(reasoner.d_seperable("Sprinkler?", "Wet grass?", "Rain?"))