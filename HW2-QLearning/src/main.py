
#%%
import numpy as np
import matplotlib.pyplot as plt

from agent import QLearningAgent
from env import BinomialStock

hedge_ratio_list = np.array([0, 0.5, 1])
n_step = 365
n_episode = 100

agent = QLearningAgent(
    n_state=n_step,
    n_action=len(hedge_ratio_list),
    learning_rate=0.1,
    gamma=0.9,
    exploration_rate=0.1,
)

env = BinomialStock(
    S0=50,
    rf=0.05/365,
    u = 1 + (0.1/365),
    n_step=n_step,
    hedge_ratio_list=hedge_ratio_list,
)

mat_S = env.sample(n_episode)

for episode in range(n_episode):
    
    S_list = mat_S[:, episode]
    for t in range(n_step):
        if t == range(n_step)[-1]:
            terminal = True
        else:
            terminal = False

        action = agent.action(t)
        hedge_ratio = hedge_ratio_list[action]
        if terminal:
            reward = 0
        else:
            reward = - np.abs(S_list[t+1] - hedge_ratio * S_list[t+1])
        agent.q_learning(t, action, reward, t+1, terminal)


print(agent.mat_Q)
# %%
