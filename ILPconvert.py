import networkx as nx
from pulp import *

def Convert_to_ILP( SFClist: list[list[nx.DiGraph]], G: nx.DiGraph) -> LpProblem:


    # define pi variable
    pi = LpVariable.dicts(
        name="pi",
        indices=(range(len(SFClist))),
        cat="Binary",
    )

    # define node variable
    xNode = LpVariable.dicts(name   = "xNode",
                            indices = ((s_index, k, i, v) 
                                for s_index, s in enumerate(SFClist)
                                for k, config in enumerate(s)
                                for i in G.nodes 
                                for v in config.nodes),
                            cat = "Binary"
                            )

    # define phi variable
    phi = LpVariable.dicts( name    ="phi",
                            indices =[(s_index, k) 
                            for s_index, s in enumerate(SFClist) 
                            for k, config in enumerate(s)],
                            cat = "Binary"
                            )

    # define edge variable
    xEdge = LpVariable.dicts(name    = "xEdge", 
                            indices  =((s_index, k, (i, j), (v, w)) 
                                for s_index, s in enumerate(SFClist)
                                for k, subgraph in enumerate(s)
                                for v, w in subgraph.edges 
                                for i, j in G.edges),
                            cat = "Binary"
                            )

    z = LpVariable.dicts(name = "z",
                        indices = [(s_index,k)
                            for s_index, s in enumerate(SFClist)
                            for k, subgraph in enumerate(s)],
                        cat = "Binary"
    )

    __problem = LpProblem(name="FIXED", sense=LpMinimize)

    # Attributes of the physical network
    aNode = nx.get_node_attributes(G, "weight")
    aEdge = nx.get_edge_attributes(G, "weight")


    for s_index, s in enumerate(SFClist):
        for k, config in enumerate(s):
            rNode = nx.get_node_attributes(config, "weight")
            rEdge = nx.get_edge_attributes(config, "weight")

            #C1:
            for i in G.nodes:
                
                __problem += (
                    lpSum(
                        xNode[s_index, k, i, v] * rNode[v]
                        for v in config.nodes
                    ) <= aNode[i] * phi[(s_index, k)],
                    f'C1_{s_index}_{k}_{i}'
                )

            
            #C2:
            for (i,j) in G.edges:
                __problem += (
                    lpSum(
                        xEdge[(s_index, k, (i, j), (v, w))] * rEdge[(v,w)]
                            for  v,w in config.edges
                    )
                    <= aEdge[(i,j)] * phi[(s_index, k)],
                    f'C2_{s_index}_{k}_{i}_{j}'
                )
            #C3:
            for i in G.nodes:
                __problem+=(
                    lpSum(
                        xNode[s_index, k, i, v]
                        for v in config.nodes
                    )
                    <= z[(s_index, k)]
                )
            
            #C4:
            for v in config.nodes:
                __problem +=(
                    lpSum(
                        xNode[s_index, k, i, v]
                        for v in config.nodes                   
                    )
                    == z[(s_index, k)]
                    
                )
            #C5:
            M = 100
            for v,w in config.edges:
                for (i,j) in G.edges:
                    __problem +=(
                        lpSum(
                            (xEdge[(s_index, k, (i, j), (v, w))] - xEdge[(s_index, k, (j, i), (v, w))]) - (xNode[s_index, k, i, v] - xNode[s_index, k, i, w])

                        )
                        <= M * (1 - phi[(s_index, k)]) 
                    )
                    __problem +=(
                        lpSum(
                            (xEdge[(s_index, k, (i, j), (v, w))] - xEdge[(s_index, k, (j, i), (v, w))]) - (xNode[s_index, k, i, v] - xNode[s_index, k, i, w])

                        )
                        >= -1 * M * (1 - phi[(s_index, k)])                        
                    )
            
            #C6:
            __problem +=(
                lpSum(
                    phi[(s_index, k)]
                )
                == 1
            )

            #C7:
            __problem += (
                z[(s_index, k)] <= pi[s_index]
            )

            __problem += (
                z[(s_index, k)] <= phi[(s_index, k)]
            )

            __problem += (
                z[(s_index, k)] >= pi[s_index] + phi[(s_index, k)] - 1
            )

            __problem += (
                0 - lpSum(
                pi[s_index]
                )
            )
    return __problem