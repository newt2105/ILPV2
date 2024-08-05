import pickle
import matplotlib.pyplot as plt
import networkx as nx

from env import RLen3
from agent import QLearningAgent, TrainAgent

PHY = nx.DiGraph()
sfcs_list = nx.DiGraph()

env = RLen3(PHY, sfcs_list)

action_space = env.action_space
observation_space = env.observation_space
agent = QLearningAgent(action_space, observation_space, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, epsilon_max=1.0)

trained_agent, rewards = TrainAgent(agent, env, nepisode=100, verbose=True, liveview=True)

# Vẽ kết quả cuối cùng
agent.plot_duration(show_result=True)

# Lưu ảnh cuối cùng
plt.savefig('final_result.png')


# Lưu mô hình Q-learning agent
with open('q_learning_agent.pkl', 'wb') as f:
    pickle.dump(agent.q_table, f)