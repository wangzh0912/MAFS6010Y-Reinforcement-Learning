#%%
from scipy.stats import norm
import numpy as np


#Create stock price series by geometric Brownian motion
class GBMStock(object):

    def __init__(self, S0, rf, sigma, n_days) -> None:
        self.S0 = S0
        self.rf = rf
        self.sigma = sigma
        self.n_step = n_days
        self.T = n_days / 365

    def sample(self, n_episode):
        self.mat_S = np.zeros((self.n_step, n_episode))
        self.mat_Call = np.zeros((self.n_step, n_episode))
        self.mat_S[0] = self.S0
        self.mat_Call[0] = self.call_price(self.S0, self.T, self.S0)
        dt = 1 / 365

        for t in range(1, self.n_step):
            tau = (self.n_step - t) / 365
            Z = np.random.standard_normal((1, n_episode))
            self.mat_S[t] = self.mat_S[t-1] * np.exp((self.rf - 0.5 * self.sigma**2) * dt + self.sigma * np.sqrt(dt) * Z)
            self.mat_Call[t] = self.call_price(self.mat_S[t], tau, self.S0) 
        return self.mat_S, self.mat_Call

    def call_price(self, s, t, k):
        d1 = (np.log(s/k) + (self.rf+0.5*self.sigma**2)*t)/(self.sigma*np.sqrt(t))
        d2 = d1 - self.sigma * np.sqrt(t)
        return norm.cdf(d1) * s - norm.cdf(d2) * k * np.exp(-self.rf * t)




    
if __name__ == '__main__':
    env = GBMStock(50, 0.05, 0.3, 10)
    stock, call = env.sample(100)
    import matplotlib.pyplot as plt
    plt.plot(stock)
# %%
