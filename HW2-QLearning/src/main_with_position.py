
#%%
import numpy as np
import matplotlib.pyplot as plt

from agent import QLearningAgent, QLearningAgentPosition
from env import BinomialStock
import numpy as np


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

def test_episode(agent, env, hedge_ratio_list, n_days, n_episode):

    mat_S = env.sample(n_episode)
    cum_reward = []

    for episode in range(n_episode):
        total_reward = 0
        S_list = mat_S[:, episode]
        state = [0, 0]  # start state
        t = state[0]  # date    

        while t != n_days - 1:   

            action = agent.predict(state)
            hedge_ratio = hedge_ratio_list[action]
            reward = - np.abs(S_list[t+1] - hedge_ratio * S_list[t+1])  # next state reward
            total_reward += reward
            next_state = [t+1,action]
            state = next_state
            t = state[0]
        
        cum_reward.append(total_reward)

    return cum_reward


def main():

    hedge_ratio_list = np.array([0, 0.5, 1])
    n_days = 365
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

    train_reward = train_episode(agent, env, hedge_ratio_list, n_days, n_episode)
    test_reward = test_episode(agent, env, hedge_ratio_list, n_days, n_episode)

    plt.plot(train_reward)
    plt.show()

    plt.plot(test_reward)
    plt.show()

if __name__ == '__main__':
    main()


# %%
