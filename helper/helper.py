import pulp as pl
import networkx as nx
from pulp import *



"""
    Arguments :
        PHY : physical network,
        K : a list of list of DiGraph represent the configuration of each slice
    Output :
        Graphmap : a LpProblem represent the graph mapping problem
"""
def ConvertToILP(PHY : nx.DiGraph,
                 K : list,
                 ) -> pl.LpProblem:
    # Create a pulp linear model
    M = 100
    gamma = 0.9
    lp_problem = pl.LpProblem(name="SE_Fixed problem",
                              sense=pl.LpMinimize)

    # x_s_k_i_v
    xNode = dict()
    for s in range(len(K)):
        xNode[s]=dict()
        for k in range(len(K[s])):
            xNode[s][k]=dict()
            for i in PHY.nodes:
                xNode[s][k][i] = pl.LpVariable.dicts(name=f"xNode_{s}_{k}_{i}",
                                                    indices= [v for v in K[s][k].nodes],
                                                    cat=pl.LpBinary)

    #x_s_k_ij_vw
    xEdge = dict()
    for s in range(len(K)):
        xEdge[s]=dict()
        for k in range(len(K[s])):
            xEdge[s][k]=dict()
            for (i,j) in PHY.edges:
                xEdge[s][k][(i,j)] = pl.LpVariable.dicts(name=f"xEdge_{s}_{k}_{(i,j)}",
                                                        indices= [(v,w) for (v,w) in K[s][k].edges],
                                                        cat=pl.LpBinary)

    #phi_s_k
    phi = dict()
    for s in range(len(K)): 
        phi[s] = pl.LpVariable.dicts(name=f"phi_{s}",
                                    indices=[k for k in range(len(K[s]))],
                                    cat=pl.LpBinary)

    #pi_s
    pi = pl.LpVariable.dicts(name="pi",
                            indices=[s for s in range(len(K))],
                            cat=pl.LpBinary)
    
    #z_s_k
    z = dict()
    for s in range(len(K)): 
        z[s] = pl.LpVariable.dicts(name=f"z_{s}",
                                    indices=[k for k in range(len(K[s]))],
                                    cat=pl.LpBinary)

    # Constraints
    # C1 : Node resources
    for i in PHY.nodes:
        lp_problem += pl.lpSum(
                    xNode[s][k][i][v]*K[s][k].nodes[v]['cap']['cpu'] 
                            for s in range(len(K))
                            for k in range(len(K[s]))
                            for v in K[s][k].nodes
                            ) <= PHY.nodes[i]['cap']['cpu'],f"C1_cpu_{i}"
        
        lp_problem += pl.lpSum(xNode[s][k][i][v]*K[s][k].nodes[v]['cap']['memory'] 
                            for s in range(len(K))
                            for k in range(len(K[s]))
                            for v in K[s][k].nodes
                            ) <= PHY.nodes[i]['cap']['memory'],f"C1_memory_{i}"
        
        lp_problem += pl.lpSum(xNode[s][k][i][v]*K[s][k].nodes[v]['cap']['storage'] 
                            for s in range(len(K))
                            for k in range(len(K[s]))
                            for v in K[s][k].nodes
                            ) <= PHY.nodes[i]['cap']['storage'],f"C1_storage_{i}"
        
    # C2 : Edge resources
    for (i,j) in PHY.edges:
        lp_problem += pl.lpSum(xEdge[s][k][(i,j)][(v,w)]*K[s][k].edges[(v,w)]["cap"]['bandwidth']
                                    for s in range(len(K))
                                    for k in range(len(K[s]))
                                    for (v,w) in K[s][k].edges
                                    ) <= PHY.edges[(i,j)]['cap']['bandwidth'],f"C2_bandwidth_{(i,j)}"
        
    # C3 : Map once
    for i in PHY.nodes:
        for s in range(len(K)):
            for k in range(len(K[s])):
                lp_problem += pl.lpSum(xNode[s][k][i][v]
                                for v in K[s][k].nodes) <= z[s][k],f"C3_{s}_{k}_{i}"

    # C4 : Map all
    for s in range(len(K)):
        for k in range(len(K[s])):
            for v in K[s][k].nodes:
                lp_problem += pl.lpSum(xNode[s][k][i][v] for i in PHY.nodes) == z[s][k],f"C4_{s}_{k}_{v}"

    # C5 : Service conservative
    for i in PHY.nodes:
        for s in range(len(K)):
            for k in range(len(K[s])):
                for (v,w) in K[s][k].edges:
                    lp_problem += pl.lpSum(xEdge[s][k][(i,j)][(v,w)] - xEdge[s][k][(j,i)][(v,w)]
                                        for j in PHY.nodes if (i,j) in PHY.edges) - (xNode[s][k][i][v] - xNode[s][k][i][w]) <= M*(1-phi[s][k]),f"C5_1_{s}_{k}_{i}_{(v,w)}"                        
                    lp_problem += pl.lpSum(xEdge[s][k][(i,j)][(v,w)] - xEdge[s][k][(j,i)][(v,w)]
                                        for j in PHY.nodes if (i,j) in PHY.edges) - (xNode[s][k][i][v] - xNode[s][k][i][w]) >= -M*(1-phi[s][k]),f"C5_2_{s}_{k}_{i}_{(v,w)}" 

    # C6 : Only one configuration
    for s in range(len(K)):
        lp_problem += pl.lpSum(phi[s][k] for k in range(len(K[s]))) == pi[s],f"C6_{s}"

    # C7 : Change variables conditions
    for s in range(len(K)):
        for k in range(len(K[s])):
            lp_problem += z[s][k] <= pi[s],f"C7_1_{k}_{s}"
            lp_problem += z[s][k] <= phi[s][k],f"C7_2_{k}_{s}"
            lp_problem += z[s][k] >= pi[s] + phi[s][k] - 1,f"C7_3_{k}_{s}"

    M = 100
    gamma = 0.9
    # Objective
    lp_problem += -pl.lpSum(pi[s] for s in range(len(K))) + (1-gamma)*(pl.lpSum(xEdge[s][k][(i,j)][(v,w)]
                                                                               for s in range(len(K)) 
                                                                               for k in range(len(K[s]))
                                                                               for (i,j) in PHY.edges
                                                                               for (v,w) in K[s][k].edges))
    

    return lp_problem, xEdge 
def extract_mapping_result(lp_problem, K, PHY,xEdge ):
    node_mapping = {}
    edge_mapping = {}
    reward = value(lp_problem.objective)
    # Extract node mappings
    for s in range(len(K)):
        for k in range(len(K[s])):
            for v in K[s][k].nodes:
                for i in PHY.nodes:
                    var_name = f"xNode_{s}_{k}_{i}_{v}"

                    var = lp_problem.variablesDict().get(var_name)
                    if var and var.varValue == 1:
                        if (s, k) not in node_mapping:
                            node_mapping[(s, k)] = {}
                        node_mapping[(s, k)][v] = i
                        break
    # print(node_mapping)
    # Extract edge mappings 
    for s in range(len(K)):
        # edge_mapping[s] = {}
        for k in range(len(K[s])):
            for phy_link in PHY.edges:
                for vlink in K[s][k].edges:
            # for vlink in K[s][k].edges:
                # for phy_link in PHY.edges:
                    a = xEdge[s][k][phy_link][vlink].name
                    var = lp_problem.variablesDict().get(a)
                    # print(var)
                    if var and var.varValue == 1:
                        if (s, k) not in edge_mapping:
                            edge_mapping[(s, k)] = {}
                            # print(edge_mapping)
                        edge_mapping[(s, k)][vlink] = phy_link
                        break
    # print(edge_mapping)
    return reward ,{'node_mapping': node_mapping, 'link_mapping': edge_mapping}



