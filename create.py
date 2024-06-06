import networkx as nx
import random

def PHY (num_node: int, node_req: list[int], edge_req: list[int]) -> nx.DiGraph:
    G = nx.DiGraph()
    for i in range(num_node):
        G.add_node(i, weight = node_req[i])
    for j in range(num_node -1):
        G.add_edge(j, j+1, weight = edge_req[j])
        G.add_edge(j+1, j, weight = edge_req[j])
    return G

def SFClist (num_slice: int , num_config: int, size: list[int], node_req: list[list[int]], edge_req: list[list[int]]) -> list[list[nx.DiGraph]]:
    SFClist = []

    for i in range(num_slice):
        sfc_config = []

        for config_index in range(num_config):
            graph = nx.DiGraph(name=f"SFC {i+1} Config {config_index+1}")
            for a in range(size[config_index]):
                graph.add_node(a, weight = node_req[config_index][a])
            for j in range(size[config_index] - 1):
                graph.add_edge(j, j+1, weight = edge_req[config_index][j])
                
        
            sfc_config.append(graph)
    
        SFClist.append(sfc_config)
    return SFClist


if __name__ == '__main__':
    num_node = 5
    node_req = [10,20,30,40,50]
    edge_req = [10,20,30,40]
    G = PHY(num_node, node_req, edge_req)
    # print(G.nodes(data= True))
    print(G.get_edge_data(0, 1)['weight'])   

    num_slice = 2
    num_config = 2
    size = [3, 4]
    node_req = [[1,2,3], [4,5,6,7]]
    edge_req = [[1,2], [3,4,5]]

    SFClist = SFClist(num_slice, num_config, size, node_req, edge_req) 

