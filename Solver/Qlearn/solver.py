import sys
import os
import time
import pickle

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


class QLearningSolver:
    def __init__(self, agent, env):
        self.agent = agent
        self.env = env

    def load_q_table(self, filepath):
        with open(filepath, 'rb') as f:
            self.agent.q_table = pickle.load(f)

    def solve(self):
        start_time = time.time()
        terminated = False
        truncated = False
        obs, info = self.env.reset()

        total_slices = len(self.env.sfcs_list)
        mapped_slices = 0

        while not terminated and not truncated:
            action = self.agent.choose_action(obs, trainmode=False)
            next_obs, reward, terminated, info = self.env.step(action)
            if reward > 0:
                mapped_slices += 1
            obs = next_obs

        end_time = time.time()
        solving_time = end_time - start_time
        accept_rate = mapped_slices / total_slices

        print(f"Solving Time: {solving_time} seconds")
        print(f"Accept Rate: {accept_rate * 100:.2f}%")
        print(f"Solution Info: {info}")

        return info