
#%%
import numpy as np
import matplotlib.pyplot as plt

from agent import QLearningAgent, QLearningAgentPosition
from env import BinomialStock
import numpy as np



hedge_ratio_list = np.array([0, 0.5, 1])
n_days = 365
n_step = n_days * len(hedge_ratio_list)  #num of state 
n_episode = 100

agent = QLearningAgentPosition(
    n_days=n_days,
    n_action=len(hedge_ratio_list),
    learning_rate=0.1,
    gamma=0.9,
    exploration_rate=0.1,
)

env = BinomialStock(
    S0=50,
    rf=0.05/365,
    u = 1 + (0.1/365),
    n_step=n_days,
    hedge_ratio_list=hedge_ratio_list,
)

mat_S = env.sample(n_episode)

for episode in range(n_episode):
    
    S_list = mat_S[:, episode]
    state = [0, 0] #start state
    t = state[0]   #date    

    while t != n_days - 1:   

        action = agent.action(state)
        hedge_ratio = hedge_ratio_list[action]
        reward = - np.abs(S_list[t+1] - hedge_ratio * S_list[t+1]) #nest state reward
        next_state = [t+1,action]
        agent.q_learning(state, action, reward, next_state)

        state = next_state
        t=state[0]


test = agent.mat_Q
# %%
