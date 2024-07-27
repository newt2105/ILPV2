from pulp import LpProblem, LpVariable, LpMaximize, COIN_CMD
import gymnasium as gym
import networkx as nx
import numpy as np
import copy

from helper.helper import extract_mapping_result, ConvertToILP

import warnings
warnings.filterwarnings("ignore", message="Spaces are not permitted in the name. Converted to '_'")

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
        self.mapped_configs = set()
        self.sol = dict()
        self.observation_space = gym.spaces.Discrete(n=self.num_sfcs + 1)
        self.action_space = gym.spaces.Discrete(3)
    
    def reset(self):
        self.physical_graph_current = copy.deepcopy(self.physical_graph)
        self.mapped_configs = set()
        self.sfc_order_current = 0
        self.sol = {}
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
        if action not in [0, 1]:
            return None  # Trả về None nếu hành động không hợp lệ
        for s_index in range(self.sfc_order_current, len(self.sfcs_list)):
            s = self.sfcs_list[s_index]
            if action < len(s):
                return (s_index, action, s[action])
        
        return None
    def _all_mapped(self):
        if self.sfc_order_current == len(self.sfcs_list):
            return True
        return False
    
    def _confirm_mapping(self):
        if self.sfc_order_current < len(self.sfcs_list):
            self.sfc_order_current += 1

    def __skip_sfc(self):
        self.sfc_order_current += 1
    
    def __is_last_slice(self):
        if self.sfc_order_current == len(self.sfcs_list):
            return True
        return False

    def __is_reached_termination(self):
        if (self.sfc_order_current >= len(self.sfcs_list)):
            return True
        return False

    

    def __update_key(self,key, sfc_index, config_index):
        if key.startswith('phi_'):
            parts = key.split('_')
            parts[1] = str(int(parts[1]) - int(parts[1]) + sfc_index)
            parts[2] = str(int(parts[2]) - int(parts[2]) + config_index)
            new_key = '_'.join(parts)
        elif key.startswith('pi_'):
            parts = key.split('_')
            parts[1] = str(int(parts[1]) - int(parts[1]) + sfc_index)
            new_key = '_'.join(parts)
        elif key.startswith('xEdge_'):
            parts = key.split('_', 3)
            parts[1] = str(int(parts[1]) - int(parts[1]) + sfc_index)
            parts[2] = str(int(parts[2]) - int(parts[2]) + config_index)
            new_key = '_'.join(parts)
        elif key.startswith('xNode_'):
            parts = key.split('_')
            parts[1] = str(int(parts[1]) - int(parts[1]) + sfc_index)
            parts[2] = str(int(parts[2]) - int(parts[2]) + config_index)
            new_key = '_'.join(parts)
        elif key.startswith('z_'):
            parts = key.split('_')
            parts[1] = str(int(parts[1]) - int(parts[1]) + sfc_index)
            parts[2] = str(int(parts[2]) - int(parts[2]) + config_index)
            new_key = '_'.join(parts)
        else:
            new_key = key
        return new_key
    
    def step(self, action):
        action = action -1
        # print("alo: ", self.action_space)
        # print("curent in step: ",self.sfc_order_current)
        if (self.__is_reached_termination()):
            reward = 0
            info = {
                "message": "the env is terminated"
            }
            
            return (self.sfc_order_current, reward, self.__is_reached_termination(), info)

        if (action == -1):
            self.__skip_sfc()
            reward = -0.3
            is_done = self._all_mapped()
            if is_done:
                info = {
                    "message": f"skip the config - ALL MAPPED"
                }
                done = True
                return self.sfc_order_current, reward, self.__is_reached_termination(), info
            else:
                done = False
                
                info = {
                    "message": "skip the sfc"
                }
                return self.sfc_order_current, reward, False, info          
        
        sfc_index, config_index, sfc = self._get_action_detail(action)
        if (sfc_index, config_index) in self.mapped_configs:
            reward = -10
            info = {
                {"message": f"config {config_index} of SFC {sfc_index} already mapped"}
            }
            return self.sfc_order_current, reward, self.__is_reached_termination(), info

        K = []
        K.append([sfc])
        
        problem, xEdge, phi, pi, z= ConvertToILP(self.physical_graph_current, K)
        solver = COIN_CMD(msg=0)  # Tạo đối tượng solver với thông số msg=0 để tắt log
        problem.solve(solver)
        new_solution = dict()
        for v in problem.variables():
            if v.varValue == 1.0:
                self.sol[v.name] = v.varValue
        
        for key, value in self.sol.items():
            if value == 1.0:
                new_key = self.__update_key(key, sfc_index, config_index)
                new_solution[new_key] = value
        self.sol.update(new_solution)
        reward, mapping_result = extract_mapping_result(problem, K, self.physical_graph, xEdge)
        reward = -reward
        info = {}
        self.update_physical_network(mapping_result, K)
        self._confirm_mapping()
        self.mapped_configs.add((sfc_index, config_index))
        info = {
            "mesage": f"config {config_index} of SFC {sfc_index} mapped successful into PHY "
        }
        done = False
        is_done = self._all_mapped()
        if is_done:
            info = {
                "message": "All SFCs mapped"
            }
            done = True
            return self.sfc_order_current, reward, self.__is_reached_termination(), info
        else:
            done = False
        
        return self.sfc_order_current, reward, self.__is_reached_termination(), info
    
    def render(self)->dict:
        return self.sol