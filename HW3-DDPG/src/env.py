#%%
from cmath import exp
from scipy.stats import norm
import numpy as np


#Create stock price series by geometric Brownian motion
class GBMStock(object):

    def __init__(self, S0, rf, sigma, T) -> None:
        self.S0 = S0
        self.rf = rf
        self.sigma = sigma
        self.T = T
        self.n_step = round(T * 365)

    def sample(self, n_episode):
        self.mat_S = np.zeros((self.n_step, n_episode))
        self.mat_Call = np.zeros((self.n_step, n_episode))
        self.mat_S[0] = self.S0
        self.mat_Call[0] = self.call_price(self.S0, self.T, self.S0)
        dt = 1 / 365

        for t in range(1, self.n_step):
            Z = np.random.standard_normal((1, n_episode))
            self.mat_S[t] = self.mat_S[t-1] * np.exp((self.rf - 0.5 * self.sigma**2) * dt + self.sigma * np.sqrt(dt) * Z)
            self.mat_Call[t] = self.call_price(self.mat_S[t],self.T-t* dt,self.S0) 
        return self.mat_S

    def call_price(self, s ,t, k):
        d1 = (np.log(s/k) + (self.rf+0.5*self.sigma**2)*t)/(self.sigma*np.sqrt(t))
        d2 = d1 - self.sigma * np.sqrt(t)
        return norm.ppf(d1)*s - norm.ppf(d2) * k *np.exp(-self.rf * t)




    
if __name__ == '__main__':
    env = GBMStock(50, 0.05, 0.3, 0.5)
    price = env.sample(10)
# %%
