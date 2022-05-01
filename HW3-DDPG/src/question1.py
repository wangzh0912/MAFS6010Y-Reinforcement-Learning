import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from agent import PolicyGradientAgent
from env import GBMStock


action_bound = [0, 1]
learning_rate = 0.1
gamma = 0.9
T = 1
env = GBMStock(50, 0.05, 0.3, T)
n_step = env.n_step

agent = PolicyGradientAgent(n_step, action_bound, learning_rate, gamma)
n_episode = 100

mat_S, mat_V = env.sample(n_episode)
cum_reward = []

episode = 1
# generate 1 episode
t_list = range(n_step)
S_list = mat_S[:, episode]
V_list = mat_V[:, episode]
act_list = np.zeros(n_step)
reward_list = np.zeros(n_step)
for t in t_list:
    state = [t_list[t], S_list[t], V_list[t]]
    act_list[t] = agent.predict(state)
    if t > 0:
        reward_list[t] = np.abs(V_list[t] - act_list[t-1] * S_list[t])


def train_episode(agent, env, hedge_ratio_list, n_days, n_episode):

    mat_S = env.sample(n_episode)
    cum_reward = []
    
    for episode in range(n_episode):
        
        total_reward = 0
        S_list = mat_S[:, episode]
        state = [0, 0]  # start state
        t = state[0]  # date    

        while t != n_days - 1:   

            action = agent.action(state)
            hedge_ratio = hedge_ratio_list[action]
            reward = - np.abs(S_list[t+1] - hedge_ratio * S_list[t+1])  # next state reward
            total_reward += reward
            next_state = [t+1,action]
            agent.q_learning(state, action, reward, next_state)
            state = next_state
            t = state[0]

        cum_reward.append(total_reward)

        if episode % 10 == 0:
            print('Finish training %d%% episodes.' % int(episode/n_episode*100))

    return cum_reward