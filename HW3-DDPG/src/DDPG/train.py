#%%
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from agent import DDPG
from env import GBMStock


hyper_params = {
    'n_train_episodes': 2000,
    'n_days': 90,
    'lr_actor': 1e-6,
    'lr_critic': 1e-6,
    'discount_rate': 0.9,
    'replacement_rate':0.01,
    'memory_size': 2000,
    'batch_size': 32,
}


env = GBMStock(50, 0.05, 0.3, hyper_params['n_days'])
state_dim = 3
act_dim = 1
agent = DDPG(state_dim, act_dim, hyper_params)


# def train
n_step = env.n_step
mat_S, mat_V = env.sample(hyper_params['n_train_episodes'])
cum_reward = []

for episode in range(hyper_params['n_train_episodes']):

    t_list = range(n_step)
    S_list = mat_S[:, episode]
    V_list = mat_V[:, episode]
    act_list = np.zeros(n_step)
    reward_list = np.zeros(n_step)

    for t in t_list:
        state = [t_list[t], S_list[t], V_list[t]]
        act_list[t] = agent.predict(state)
        # TODO 随机性

        if t > 0:
            reward_list[t] = -np.abs(V_list[t] - act_list[t-1] * S_list[t])

        if t < n_step-1:
            state_next = [t_list[t+1], S_list[t+1], V_list[t+1]]
            agent.store_memory(state, act_list[t], reward_list[t], state_next)

        
        if agent.pointer > hyper_params['memory_size']:
            agent.learn()
        
    cum_reward.append(np.sum(reward_list))

    if episode/hyper_params['n_train_episodes']*100 % 10 == 0:
        print('Finish training %d%% episodes.' % int(episode/hyper_params['n_train_episodes']*100))

plt.plot(range(hyper_params['n_train_episodes']), cum_reward)
plt.show()
# %%
