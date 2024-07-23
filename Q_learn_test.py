import matplotlib
import networkx as nx
import matplotlib.pyplot as plt
import torch
from IPython import display
from collections import defaultdict
import numpy as np
from env import RLen3

        
# set up matplotlib
IS_IPYTHON = 'inline' in matplotlib.get_backend()
if IS_IPYTHON:
    from IPython import display
plt.ion()

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

class QLearningAgent:
    def __init__(self, action_space, observation_space, learning_rate=0.1, discount_factor=0.99, epsilon=0.1, epsilon_min=0.01, epsilon_decay=0.995, epsilon_max=1.0):
        self.action_space = action_space
        self.observation_space = observation_space
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.epsilon_max = epsilon_max
        self.q_table = defaultdict(lambda: np.zeros(action_space.n))
        self.episode_duration = []

    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            return self.action_space.sample()
        else:
            return np.argmax(self.q_table[state])

    def update_q_table(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.discount_factor * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.learning_rate * td_error

    def end_episode(self, reset=False):
        if reset:
            self.epsilon = self.epsilon_max
        else:
            new_epsilon = self.epsilon * self.epsilon_decay
            self.epsilon = max(new_epsilon, self.epsilon_min)

    def plot_duration(self, show_result=False):
        plt.figure(1)
        duration_t = torch.tensor(self.episode_duration, dtype=torch.float)
        if show_result:
            plt.title("QL-Result")
        else:
            plt.clf()
            plt.title("QL-Training")
        plt.xlabel("Episode")
        plt.ylabel("Cumulative reward")
        plt.plot(duration_t.numpy(), color='silver')  # plot cumulative reward
        if len(duration_t) >= 100:
            means = duration_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy(), color='k')  # plot average reward
        plt.pause(0.001)
        if IS_IPYTHON:
            if not show_result:
                display.display(plt.gcf())
                display.clear_output(wait=True)
            else:
                display.display(plt.gcf())

def TrainAgent(agent: QLearningAgent, env: RLen3, nepisode: int, verbose: bool = False, liveview: bool = False) -> tuple[QLearningAgent, list[float]]:
    reward_list = []
    for ep in range(nepisode):
        obs, info = env.reset()
        terminated = False
        rw_list = []
        while not terminated:
            action = agent.choose_action(obs)
            next_obs, reward, done, info = env.step(action)
            rw_list.append(reward)
            agent.update_q_table(obs, action, reward, next_obs)
            obs = next_obs
            print(obs)
            terminated = done
        if verbose:
            print(f"ep_{ep}: {info['message']} {obs} {info}")
        agent.end_episode()
        rw = sum(rw_list)  # cumulative reward
        reward_list.append((ep, rw))
        if liveview:
            agent.episode_duration.append(rw)
            agent.plot_duration()
    return agent, reward_list

env = RLen3(PHY, sfcs_list)

action_space = env.action_space
observation_space = env.observation_space
agent = QLearningAgent(action_space, observation_space, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, epsilon_max=1.0)

trained_agent, rewards = TrainAgent(agent, env, nepisode=1000, verbose=True, liveview=True)

# Vẽ kết quả cuối cùng
agent.plot_duration(show_result=True)
