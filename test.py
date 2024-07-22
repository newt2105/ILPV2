from pulp import LpProblem, LpVariable, LpMaximize, COIN_CMD
import gymnasium as gym
import networkx as nx
import numpy as np
import copy

from helper import extract_mapping_result, ConvertToILP


# Tạo đồ thị vật lý (physical graph)
PHY = nx.DiGraph()
PHY.add_node(0, cap={'cpu': 100, 'memory': 160, 'storage': 100})
PHY.add_node(1, cap={'cpu': 100, 'memory': 160, 'storage': 100})
PHY.add_node(2, cap={'cpu': 100, 'memory': 160, 'storage': 100})
PHY.add_node(3, cap={'cpu': 100, 'memory': 160, 'storage': 100})
PHY.add_edge(0, 1, cap={'bandwidth': 100})
PHY.add_edge(1, 0, cap={'bandwidth': 100})
PHY.add_edge(1, 2, cap={'bandwidth': 100})
PHY.add_edge(2, 1, cap={'bandwidth': 100})
PHY.add_edge(2, 0, cap={'bandwidth': 100})
PHY.add_edge(0, 2, cap={'bandwidth': 100})
PHY.add_edge(2, 3, cap={'bandwidth': 100})
PHY.add_edge(3, 2, cap={'bandwidth': 100})

# Tạo các SFCs (Service Function Chains)
sfc1_config1 = nx.DiGraph()
sfc1_config1.add_node(0, cap={'cpu': 10, 'memory': 10, 'storage': 10})
sfc1_config1.add_node(1, cap={'cpu': 10, 'memory': 10, 'storage': 10})
sfc1_config1.add_node(2, cap={'cpu': 10, 'memory': 10, 'storage': 10})
sfc1_config1.add_node(3, cap={'cpu': 2, 'memory': 4, 'storage': 10})
sfc1_config1.add_edge(0, 1, cap={'bandwidth': 10})
sfc1_config1.add_edge(1, 2, cap={'bandwidth': 10})
sfc1_config1.add_edge(2, 3, cap={'bandwidth': 10})

sfc1_config2 = nx.DiGraph()
sfc1_config2.add_node(0, cap={'cpu': 10, 'memory': 10, 'storage': 10})
sfc1_config2.add_node(1, cap={'cpu': 5, 'memory': 5, 'storage': 5})
sfc1_config2.add_node(2, cap={'cpu': 20, 'memory': 20, 'storage': 20})
sfc1_config2.add_node(3, cap={'cpu': 5, 'memory': 5, 'storage': 5})
sfc1_config2.add_edge(0, 1, cap={'bandwidth': 10})
sfc1_config2.add_edge(1, 2, cap={'bandwidth': 15})
sfc1_config2.add_edge(3, 2, cap={'bandwidth': 15})
sfc1_config2.add_edge(0, 3, cap={'bandwidth': 10})

# Tạo danh sách các SFCs cho 3 slice, mỗi slice có 2 cấu hình
sfcs_list = [[sfc1_config1, sfc1_config2] for _ in range(5)]
# print(len(sfcs_list))
# exit()

# Định nghĩa môi trường RLen3
class RLen3(gym.Env):
    action_space = gym.Space()
    observation_space = gym.Space() 
    num_config = int()
    physical_graph = nx.DiGraph()
    
    def __init__(self, physical_graph: nx.DiGraph, sfcs_list: list[list[nx.DiGraph]]):
        super(RLen3, self).__init__() 
        self.physical_graph = physical_graph
        self.sfcs_list = sfcs_list
        self.num_sfcs = len(sfcs_list)
        # self.num_nodes = len(physical_graph.nodes)
        # self.num_links = len(physical_graph.edges)
        self.mapped_configs = 0
        
        self.observation_space = gym.spaces.Discrete(n=self.num_sfcs)
        self.action_space = gym.spaces.Discrete(n=sum(len(s) for s in sfcs_list))
    
    def reset(self):
        self.physical_graph_current = copy.deepcopy(self.physical_graph)
        self.mapped_configs = 0
        observation = copy.deepcopy(self.physical_graph)
        self.sfc_order_current = 0
        return (self.sfc_order_current, {"message": "environment reset"})
        
    def __get_node_cap(self, node_id): 
        node_caps = nx.get_node_attributes(self.physical_graph_current, "cap")
        if node_id is not None:
            return node_caps.get(node_id, {})
        return node_caps
    
    def __get_link_cap(self, link_id):
        link_caps = nx.get_edge_attributes(self.physical_graph_current, "cap")
        if link_id is not None:
            return link_caps.get(link_id, {})
        return link_caps
    
    def __get_vnode_req(self, sfc, vnf_id):  
        vnf_reqs = nx.get_node_attributes(sfc, "cap")
        if vnf_id is not None:
            return vnf_reqs.get(vnf_id, {})
        return vnf_reqs

    def __get_vlink_req(self, sfc, vlink_id):
        vlink_reqs = nx.get_edge_attributes(sfc, "cap")
        if vlink_id is not None:
            return vlink_reqs.get(vlink_id, {})
        return vlink_reqs

    def update_physical_network(self, mapping_result, K):
        node_mapping = mapping_result.get('node_mapping', {})
        link_mapping = mapping_result.get('link_mapping', {})
        
        for sfc_id, vnf in node_mapping.items():
            for vnf_id, phy_node in vnf.items():
                vnode_req = self.__get_vnode_req(K[sfc_id[0]][sfc_id[1]], vnf_id)
                node_cap = self.__get_node_cap(phy_node)
                
                if node_cap and vnode_req:
                    updated_cap = {
                        'cpu': node_cap['cpu'] - vnode_req['cpu'],
                        'memory': node_cap['memory'] - vnode_req['memory'],
                        'storage': node_cap['storage'] - vnode_req['storage']
                    }
                    nx.set_node_attributes(self.physical_graph_current, {phy_node: updated_cap}, "cap")
        
        for sfc_id, vlink in link_mapping.items():
            for vlink_id, phylink in vlink.items():
                vlink_req = self.__get_vlink_req(K[sfc_id[0]][sfc_id[1]], vlink_id)
                link_cap = self.__get_link_cap(phylink)
                
                if link_cap and vlink_req:
                    updated_cap = {
                        'bandwidth': link_cap['bandwidth'] - vlink_req['bandwidth']
                    }
                    nx.set_edge_attributes(self.physical_graph_current, {phylink: updated_cap}, "cap")
                    
    def _get_action_detail(self, action):
        count = 0
        for s_index, s in enumerate(self.sfcs_list):
            for k, config in enumerate(s):
                if count == action:
                    return (s_index, k, config)
                count += 1
    
    def _all_mapped(self):
        
        if self.sfc_order_current == len(self.sfcs_list):
            return True
        return False
    
    def _confirm_mapping(self):
        self.sfc_order_current += 1
    
    def step(self, action):
        sfc_index, config_index, sfc = self._get_action_detail(action)
        K = []
        K.append([sfc])
        problem, xEdge = ConvertToILP(self.physical_graph_current, K)
        solver = COIN_CMD(msg=0)  # Tạo đối tượng solver với thông số msg=0 để tắt log
        problem.solve(solver)
        
        reward, mapping_result = extract_mapping_result(problem, K, self.physical_graph, xEdge)
        self.update_physical_network(mapping_result, K)
        
        is_done = self._all_mapped()
        info = {}
        if is_done:
            info = {
                "message": "All SFCs mapped"
            }
            done = True
        else:
            done = False
        
        self._confirm_mapping()
        
        
        return self.sfc_order_current, reward, done, info

# Tạo môi trường
env = RLen3(PHY, sfcs_list)

# Đặt lại môi trường
observation = env.reset()
print("Initial observation:", observation)

# Thực hiện các bước hành động để kiểm tra

while True:
    actions = [0, 1]
    action = np.random.choice(actions)  # Sử dụng numpy để chọn ngẫu nhiên một giá trị từ actions
    print(action)
    observation, reward, done, info = env.step(action)
    print(f"Action {action}:")
    print("Observation:", observation)
    print("Reward:", reward)
    print("Done:", done)
    print("Info:", info)
    
    if done:  # Thêm điều kiện để thoát khỏi vòng lặp khi done là True
        break

# Kiểm tra trạng thái cuối cùng của đồ thị vật lý
print("Final physical graph state (nodes):", env.physical_graph_current.nodes(data=True))
print("Final physical graph state (edges):", env.physical_graph_current.edges(data=True))

#Viet Q_learn o day