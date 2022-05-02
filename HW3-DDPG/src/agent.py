#%%
import numpy as np
from scipy import stats


class PolicyGradientAgent(object):

    def __init__(self, n_step, action_bound, learning_rate, gamma) -> None:
        self.n_step = n_step
        self.action_bound = action_bound
        self.learning_rate = learning_rate
        self.gamma = gamma

        self.num_state = 3
        
        self.theta = np.zeros((2 * self.num_state, 1))

    def feature_vec(self, state):
        return state
        # return stats.zscore(state)
        return state / 365


    def predict(self, state):

        # state = [time, stock price, call price]

        state = np.array(state).reshape((self.num_state, 1))
        feature = self.feature_vec(state)
        theta_mu = self.theta[:self.num_state].reshape((self.num_state, 1))
        theta_sigma = self.theta[self.num_state:].reshape((self.num_state, 1))

        mu = (theta_mu.T @ feature)[0, 0]
        sigma = np.exp((theta_sigma.T @ feature)[0, 0])

        action = np.random.normal(loc=mu, scale=sigma)

        if action >= self.action_bound[1]:
            action = self.action_bound[1]
        elif action <= self.action_bound[0]:
            action = self.action_bound[0]

        return action


    def learn(self, state_curr, act_curr, reward_total):

        # state = [time, stock price, call price]

        state_idx = state_curr[0]
        state_curr = np.array(state_curr).reshape((self.num_state, 1))
        feature = self.feature_vec(state_curr)
        theta_mu = self.theta[:self.num_state].reshape((self.num_state, 1))
        theta_sigma = self.theta[self.num_state:].reshape((self.num_state, 1))

        mu = (theta_mu.T @ feature)[0, 0]
        sigma = np.exp((theta_sigma.T @ feature)[0, 0])

        gradient_ln_mu = 1 / (sigma**2) * (act_curr - mu) * feature
        gradient_ln_sigma = ((act_curr - mu)**2 / (sigma**2) - 1) * feature
        gradient_ln = (np.concatenate([gradient_ln_mu, gradient_ln_sigma])).reshape(2*self.num_state, 1)

        self.theta = self.theta + self.learning_rate * (self.gamma**state_idx) * reward_total * gradient_ln



agent = PolicyGradientAgent(100, [0, 1], 0.1, 0.1)
agent.learn((1, 20, 3), 0.5, 23)

agent.predict([0, 50, 4])
# %%
