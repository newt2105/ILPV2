import networkx as nx
import random
from pulp import *

from ILPconvert import Convert_to_ILP
from create import PHY, SFClist


def main():
    num_node = 5
    node_req = [10,20,30,40,50]
    edge_req = [10,20,30,40]
    G = PHY(num_node, node_req, edge_req)
    # print(G.nodes(data= True))
    print(G.edges[0,1]["weight"])   

    num_slice = 2
    num_config = 2
    size = [3, 4]
    node_req = [[1,2,3], [4,5,6,7]]
    edge_req = [[1,2], [3,4,5]]

    SFC_list = SFClist(num_slice, num_config, size, node_req, edge_req) 
    # print(SFC_list) 

    problem = Convert_to_ILP(SFC_list, G)
    # Giải quyết bài toán
    problem.solve()

    # In kết quả
    print(f"Status: {LpStatus[problem.status]}")
    for v in problem.variables():
        print(f"{v.name} = {v.varValue}")

    print(f"Objective value: {value(problem.objective)}")

if __name__ == '__main__':
    main()
