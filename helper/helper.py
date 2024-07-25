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



# # Create a physical network graph (PHY)
# # PHY = nx.DiGraph()
# # PHY.add_node(0, cap={'cpu': 100, 'memory': 160, 'storage': 100})
# # PHY.add_node(1, cap={'cpu': 100, 'memory': 160, 'storage': 100})
# # PHY.add_node(2, cap={'cpu': 100, 'memory': 160, 'storage': 100})
# # PHY.add_node(3, cap={'cpu': 100, 'memory': 160, 'storage': 100})
# # PHY.add_edge(0, 1, cap={'bandwidth': 100})
# # PHY.add_edge(1, 0, cap={'bandwidth': 100})
# # PHY.add_edge(1, 2, cap={'bandwidth': 100})
# # PHY.add_edge(2, 1, cap={'bandwidth': 100})
# # PHY.add_edge(2, 0, cap={'bandwidth': 100})
# # PHY.add_edge(0, 2, cap={'bandwidth': 100})
# # PHY.add_edge(2, 3, cap={'bandwidth': 100})
# # PHY.add_edge(3, 2, cap={'bandwidth': 100})
# PHY = nx.DiGraph()
# PHY.add_node(0, cap={'cpu': 100, 'memory': 160, 'storage': 100})
# PHY.add_node(1, cap={'cpu': 100, 'memory': 160, 'storage': 100})
# PHY.add_node(2, cap={'cpu': 100, 'memory': 160, 'storage': 100})
# PHY.add_node(3, cap={'cpu': 100, 'memory': 160, 'storage': 100})
# PHY.add_edge(0, 1, cap={'bandwidth': 100})
# PHY.add_edge(1, 0, cap={'bandwidth': 100})
# PHY.add_edge(1, 2, cap={'bandwidth': 100})
# PHY.add_edge(2, 1, cap={'bandwidth': 100})
# PHY.add_edge(2, 0, cap={'bandwidth': 100})
# PHY.add_edge(0, 2, cap={'bandwidth': 100})
# PHY.add_edge(2, 3, cap={'bandwidth': 100})
# PHY.add_edge(3, 2, cap={'bandwidth': 100})
# # Create service function chains (SFCs) with two configurations each
# K = []

# # SFC 1 with two configurations
# sfc1_config1 = nx.DiGraph()
# sfc1_config1.add_node(0, cap={'cpu': 10, 'memory': 10, 'storage': 10})
# sfc1_config1.add_node(1, cap={'cpu': 10, 'memory': 10, 'storage': 10})
# sfc1_config1.add_node(2, cap={'cpu': 10, 'memory': 10, 'storage': 10})
# sfc1_config1.add_node(3, cap={'cpu': 2, 'memory': 4, 'storage': 10})
# sfc1_config1.add_edge(0, 1, cap={'bandwidth': 10})
# sfc1_config1.add_edge(1, 2, cap={'bandwidth': 10})
# sfc1_config1.add_edge(2, 3, cap={'bandwidth': 10})

# sfc1_config2 = nx.DiGraph()
# sfc1_config2.add_node(0, cap={'cpu': 10, 'memory': 10, 'storage': 10})
# sfc1_config2.add_node(1, cap={'cpu': 5, 'memory': 5, 'storage': 5})
# sfc1_config2.add_node(2, cap={'cpu': 20, 'memory': 20, 'storage': 20})
# sfc1_config2.add_node(3, cap={'cpu': 5, 'memory': 5, 'storage': 5})
# sfc1_config2.add_edge(0, 1, cap={'bandwidth': 10})
# sfc1_config2.add_edge(1, 2, cap={'bandwidth': 15})
# sfc1_config2.add_edge(3, 2, cap={'bandwidth': 15})
# sfc1_config2.add_edge(0, 3, cap={'bandwidth': 10})

# K.append([sfc1_config2])


# # Now you can use these graphs to run the function
# # lp_problem, xEdge = ConvertToILP(PHY, K)


# # print(lp_problem)
# # Solve the ILP problem
# # lp_problem.solve()
# # # # In kết quả
# # # print("Status:", LpStatus[problem.status])
# # for v in lp_problem.variables():
# #     print(v.name, "=", v.varValue)
# # reward = value(lp_problem.objective)
# # print(-reward)

# def extract_mapping_result(lp_problem, K, PHY,xEdge ):
#     node_mapping = {}
#     edge_mapping = {}
#     reward = value(lp_problem.objective)
#     # Extract node mappings
#     for s in range(len(K)):
#         for k in range(len(K[s])):
#             for v in K[s][k].nodes:
#                 for i in PHY.nodes:
#                     var_name = f"xNode_{s}_{k}_{i}_{v}"

#                     var = lp_problem.variablesDict().get(var_name)
#                     if var and var.varValue == 1:
#                         if (s, k) not in node_mapping:
#                             node_mapping[(s, k)] = {}
#                         node_mapping[(s, k)][v] = i
#                         break
#     # print(node_mapping)
#     # Extract edge mappings 
#     for s in range(len(K)):
#         # edge_mapping[s] = {}
#         for k in range(len(K[s])):
#             for phy_link in PHY.edges:
#                 for vlink in K[s][k].edges:
#             # for vlink in K[s][k].edges:
#                 # for phy_link in PHY.edges:
#                     a = xEdge[s][k][phy_link][vlink].name
#                     var = lp_problem.variablesDict().get(a)
#                     # print(var)
#                     if var and var.varValue == 1:
#                         if (s, k) not in edge_mapping:
#                             edge_mapping[(s, k)] = {}
#                             # print(edge_mapping)
#                         edge_mapping[(s, k)][vlink] = phy_link
#                         break
#     # print(edge_mapping)
#     return reward ,{'node_mapping': node_mapping, 'link_mapping': edge_mapping}

# # Extract and print the mapping results
# # node_mapping, edge_mapping = extract_mapping_result(lp_problem, K, PHY, xEdge)


# def get_node_cap(graph, node_id=None): 
#     node_caps = nx.get_node_attributes(graph, "cap")
#     if node_id is not None:
#         return node_caps.get(node_id, {})
#     return node_caps
# def get_vnode_req(sfc, vnf_id=None):  
#     vnf_reqs = nx.get_node_attributes(sfc, "cap")
#     if vnf_id is not None:
#         return vnf_reqs.get(vnf_id, {})
#     return vnf_reqs
# def get_link_cap(graph, link_id=None):
#     link_caps = nx.get_edge_attributes(graph, "cap")
#     if link_id is not None:
#         return link_caps.get(link_id, {})
#     return link_caps

# def get_vlink_req(sfc, vlink_id=None):
#     vlink_reqs = nx.get_edge_attributes(sfc, "cap")
#     if vlink_id is not None:
#         return vlink_reqs.get(vlink_id, {})
#     return vlink_reqs
# # print(get_node_cap(PHY))
# # print(get_node_cap(PHY,0)['cpu'])

# # print(get_vnode_req(K[0][0]))
# # print(get_vnode_req(K[0][0], 0))
# # print(get_link_cap(PHY))
# # print(get_link_cap(PHY, (0, 1)))
# # print(get_vlink_req(K[0][0]))
# # print(get_vlink_req(K[0][0], (0, 1)))
# def update_physical_network(G, mapping_result):
#     # Update the physical network based on the mapping result
#     node_mapping = mapping_result.get('node_mapping', {})
#     link_mapping = mapping_result.get('link_mapping', {})
    
#     # Update node capacities
#     for sfc_id, vnf in node_mapping.items():
#         for vnf_id, phy_node in vnf.items():
#             vnode_req = get_vnode_req(K[sfc_id[0]][sfc_id[1]], vnf_id)
#             node_cap = get_node_cap(G, phy_node)
            
#             if node_cap and vnode_req:
#                 updated_cap = {
#                     'cpu': node_cap['cpu'] - vnode_req['cpu'],
#                     'memory': node_cap['memory'] - vnode_req['memory'],
#                     'storage': node_cap['storage'] - vnode_req['storage']
#                 }
#                 nx.set_node_attributes(G, {phy_node: updated_cap}, "cap")
    
#     # Update link capacities
#     for sfc_id, vlink in link_mapping.items():
#         for vlink_id, phylink in vlink.items():
#             vlink_req = get_vlink_req(K[sfc_id[0]][sfc_id[1]], vlink_id)
#             link_cap = get_link_cap(G, phylink)
            
#             if link_cap and vlink_req:
#                 updated_cap = {
#                     'bandwidth': link_cap['bandwidth'] - vlink_req['bandwidth']
#                 }
#                 nx.set_edge_attributes(G, {phylink: updated_cap}, "cap")
# # reward, mapping_result = extract_mapping_result(lp_problem, K, PHY, xEdge)

# # node_mapping = mapping_result.get('node_mapping', {})
# # a = update_physical_network(PHY,mapping_result)

# # # In ra trọng số các nút
# # for node in PHY.nodes(data=True):
# #     print(f"Nút {node[0]} có trọng số là {node[1]['cap']}")
    
# # # In ra trọng số các cạnh
# # print("\nTrọng số các cạnh:")
# # for edge in PHY.edges(data=True):
# #     print(f"Cạnh từ {edge[0]} đến {edge[1]} có trọng số là {edge[2]['cap']}")