#%%


import numpy as np

class QLearningAgent():

    def __init__(self, n_state, n_action, learning_rate, gamma, exploration_rate):
        self.n_state = n_state
        self.n_action = n_action
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.exploration_rate = exploration_rate
        self.mat_Q = np.zeros((n_state, n_action))


    def action(self, state):
        if np.random.uniform(0, 1) < self.exploration_rate:  # explore
            action = np.random.choice(self.n_action)
        else:  # exploitation
            action = self.predict(state)
        return action

    def predict(self, state):
        Q_current = self.mat_Q[state, :]
        Q_max = np.max(Q_current)
        action_list = np.where(Q_current == Q_max)[0]
        action = np.random.choice(action_list)
        return action

    def q_learning(self, state_curr, action_curr, reward, state_next, terminal):
        Q_old = self.mat_Q[state_curr, action_curr]
        if terminal:
            Q_new = reward
        else:
            Q_new = reward + self.gamma * np.max(self.mat_Q[state_next, :])
        self.mat_Q[state_curr, action_curr] += self.learning_rate * (Q_new - Q_old)

agent = QLearningAgent(n_state=300, n_action=3, learning_rate=0.01, gamma=0.9, exploration_rate=0.1)
agent.action(3)

# %%
