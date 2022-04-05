import numpy as np

class QLearningAgent(object):

    def __init__(self, n_state, n_action, learning_rate, gamma, exploration_rate) -> None:
        self.n_state = n_state
        self.n_action = n_action
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.exploration_rate = exploration_rate
        self.mat_Q = np.zeros((n_state, n_action))  #Q function

    # take action based on the epsilon-greedy
    def action(self, state):
        if np.random.uniform(0, 1) < self.exploration_rate:  # explore
            action = np.random.choice(self.n_action)
        else:  # exploitation
            action = self.predict(state)
        return action

    # select action which maximized q funciton
    def predict(self, state):
        Q_current = self.mat_Q[state, :]
        Q_max = np.max(Q_current)
        action_list = np.where(Q_current == Q_max)[0]
        action = np.random.choice(action_list)
        return action

    #episode with Q learning
    def q_learning(self, state_curr, action_curr, reward, state_next, terminal):
        Q_old = self.mat_Q[state_curr, action_curr]
        if terminal:
            Q_new = reward
        else:
            Q_new = reward + self.gamma * np.max(self.mat_Q[state_next, :])
        self.mat_Q[state_curr, action_curr] += self.learning_rate * (Q_new - Q_old)


class QLearningAgentPosition(object):

    def __init__(self, n_days, n_action, learning_rate, gamma, exploration_rate) -> None:
        self.n_days = n_days
        self.n_action = n_action 
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.exploration_rate = exploration_rate
        self.mat_Q = np.zeros((n_days, n_action, n_action))  #Q function 3D
        self.n_state = n_days * n_action

    # take action based on the epsilon-greedy
    def action(self, state):
        if np.random.uniform(0, 1) < self.exploration_rate:  # explore
            action = np.random.choice(self.n_action)
        else:  # exploit
            action = self.predict(state)
        return action

    # select action which maximized q funciton
    def predict(self, state):
        Q_current = self.mat_Q[state[0], state[1], : ]
        Q_max = np.max(Q_current)
        action_list = np.where(Q_current == Q_max)[0]
        action = np.random.choice(action_list)
        return action

    #episode with Q learning
    def q_learning(self, state_curr, action_curr, reward, state_next, terminal):
        Q_old = self.mat_Q[state_curr[0], state_curr[1], action_curr]
        if terminal:
            Q_new = reward
        else:
            Q_new = reward + self.gamma * np.max(self.mat_Q[state_next[0], state_next[1], :])
        self.mat_Q[state_curr[0], state_curr[1], action_curr] += self.learning_rate * (Q_new - Q_old)



