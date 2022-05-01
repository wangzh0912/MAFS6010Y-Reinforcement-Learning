#%%
import numpy as np

class PolicyGradientAgent(object):

    def __init__(self, n_step, action_bound, learning_rate, gamma) -> None:
        self.n_step = n_step
        self.action_bound = action_bound
        self.learning_rate = learning_rate
        self.gamma = gamma

        
        init_mu = np.zeros((n_step, 2)) + 0.5   # initialize mean as 0.5
        init_sigma = np.zeros((n_step, 2)) + 1  # initialize std as 1
        
        self.mat_theta = np.concatenate([init_mu, init_sigma], axis=1)  # mat_theta is (n_step * 4)


    # select action which maximized q funciton
    def predict(self, state):

        state_idx = state[0]
        state = np.array(state).reshape((2, 1))
        feature = state.copy()
        theta_mu = self.mat_theta[state_idx][:2].reshape((2, 1))
        theta_sigma = self.mat_theta[state_idx][2:].reshape((2, 1))

        mu = (theta_mu.T @ feature)[0, 0]
        sigma = (theta_sigma.T @ feature)[0, 0]

        action = np.random.normal(loc=mu, scale=sigma)
        if action >= self.action_bound[1]:
            action = self.action_bound[1]
        elif action <= self.action_bound[0]:
            action = self.action_bound[0]

        return action


    def learn(self, state_curr, act_curr, reward_total):

        state_idx = state_curr[0]
        state_curr = np.array(state_curr).reshape((2, 1))
        feature = state_curr.copy()
        theta_mu = self.mat_theta[state_idx][:2].reshape((2, 1))
        theta_sigma = self.mat_theta[state_idx][2:].reshape((2, 1))

        mu = (theta_mu.T @ feature)[0, 0]
        sigma = (theta_sigma.T @ feature)[0, 0]

        gradient_ln_mu = 1 / (sigma**2) * (act_curr - mu) * feature
        gradient_ln_sigma = ((act_curr - mu)**2 / (sigma**2) - 1) * feature
        gradient_ln = (np.concatenate([gradient_ln_mu, gradient_ln_sigma])).reshape(1, 4)

        self.mat_theta[state_idx] = self.mat_theta[state_idx] + self.learning_rate * (self.gamma**state_idx) * reward_total * gradient_ln



agent = PolicyGradientAgent(100, [0, 1], 0.1, 0.1)
agent.learn((1, 2), 0.5, 23)

# %%
