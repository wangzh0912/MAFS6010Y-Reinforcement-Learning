#%%
import numpy as np


#Create stock price series
class BinomialStock(object):

    def __init__(self, S0, rf, u, n_step, hedge_ratio_list) -> None:
        self.S0 = S0
        self.R = 1 + rf
        self.u = u
        self.d = 1 / u
        self.n_step = n_step
        self.p = (self.R - self.d) / (self.u - self.d)
        self.hedge_ratio_list = hedge_ratio_list
    

    def sample(self, n_episode):
        self.mat_S = np.zeros((self.n_step, n_episode))
        self.mat_S[0] = self.S0

        for i in range(1, self.n_step):
            prob_up = np.random.binomial(1, self.p, size=n_episode)
            self.mat_S[i] = self.mat_S[i-1] * prob_up * self.u + self.mat_S[i-1] * (1-prob_up) * self.d

        return self.mat_S


