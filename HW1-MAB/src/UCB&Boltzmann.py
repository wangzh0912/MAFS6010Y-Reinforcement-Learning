#%%
from os.path import join
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn')
from tools.project_path import *

df_close = pd.read_csv(join(cleaned_data_path, 'US_price.csv'), index_col=0)
df_close.index = pd.to_datetime(df_close.index)
df_price = df_close.iloc[:, 0:20]
df_ret = df_price.pct_change()
df_ret = df_ret.dropna()
n_arm= len(df_price.columns)

#n_operation[:] = len(df_ret[('2016-01-01' <= df_ret.index ) & (df_ret.index <= '2016-01-31')])
initial_reward_estimate = df_ret[('2016-01-01' <= df_ret.index ) & (df_ret.index <= '2016-01-31')].mean()
initial_reward_estimate = np.array(initial_reward_estimate)

df_bt_ret = df_ret[df_ret.index > '2016-01-31']
mat_bt_ret = np.array(df_bt_ret)   #test data

#%%
def ucb(mat_bt_ret,c,n_arm):

    curr_reward_estimate = initial_reward_estimate
    cum_reward=1
    n_operation = np.zeros(n_arm)
    cum_reward_series=[]

    for t in range(len(mat_bt_ret)):

        max_upper_bound = 0
    
        for j in range(n_arm):
            #iterate all arms
        
            if (n_operation[j] > 0):
                #calculate bound range        
                delta_t = np.sqrt(c * np.log(t + 1)/n_operation[j])
            
                # calculate upper bound
                upper_bound = curr_reward_estimate[j] + delta_t          
            else:           
                # initialize upper bound
                upper_bound = 100           
            if upper_bound > max_upper_bound:
            
                # update max UCB
                max_upper_bound = upper_bound
            
                best_arm = j

        #find reward of best arm j  
        best_arm_reward= mat_bt_ret[t,best_arm]
        #update j's estimate
        curr_reward_estimate[best_arm] = (curr_reward_estimate[best_arm] * n_operation[best_arm] + best_arm_reward)/(n_operation[best_arm] + 1)
    
        # update number of opernation for the current arm j
        n_operation[best_arm] += 1
            
        # cumulate reward
        cum_reward*=(1+ best_arm_reward)
        cum_reward_series.append(cum_reward)

    return  cum_reward_series,cum_reward,n_operation


def boltzmann(n_arm,initial_reward_estimate,mat_bt_ret, sigma):
    n_operation = np.zeros(n_arm)
    curr_reward_estimate = initial_reward_estimate
    cum_reward=1
    cum_reward_series=[]
    
    for t in range(len(mat_bt_ret)):
        
        # probability of arm to be selected based on the current estimated reward and Boltzmann distribution
        reward_prob = np.exp(np.array(curr_reward_estimate)/sigma)/np.exp(np.array(curr_reward_estimate)/sigma).sum()
        
        # select arm based on previous probability distrivution
        best_arm = np.random.choice(n_arm, size=1, p=reward_prob)[0]
        best_arm_reward = mat_bt_ret[t,best_arm]
  
        curr_reward_estimate[best_arm] = (curr_reward_estimate[best_arm] * n_operation[best_arm] + best_arm_reward)/(n_operation[best_arm] + 1)
        n_operation[best_arm] += 1
        cum_reward*=(1+ best_arm_reward)  
        cum_reward_series.append(cum_reward)

    return cum_reward_series,cum_reward, n_operation   
#%%
#run backtesting for Ucb 
c=2
ubc_temp=pd.DataFrame(np.zeros(len(mat_bt_ret)))

temp_list = []

temp, re,n_operation = ucb(mat_bt_ret, c, n_arm)
temp = pd.DataFrame(temp)
temp.columns = [i]
temp_list.append(temp)

ubc_temp = pd.concat(temp_list, axis=1)
ubc_temp.index = df_bt_ret.index
plt.plot(ubc_temp)
plt.xlabel('time', fontsize=12)
plt.ylabel('cumulative return', fontsize=12)
plt.show()

re=6.75-re #6.75 is the best arm(aapl)'s reward
print(re) #cum_reward

#%%
#run backtest for boltzmann
N=100  #number of path
bol_temp=pd.DataFrame(np.zeros(len(mat_bt_ret)))
bol_temp.index = df_bt_ret.index
temp_list = []
re_bol = []
sigma=0.05

for i in range(N):
    temp,re, n_operation = boltzmann(n_arm, initial_reward_estimate,mat_bt_ret, sigma)
    temp = pd.DataFrame(temp)
    temp.columns = [i]
    temp_list.append(temp)
    re_bol.append(re)

bol_temp = pd.concat(temp_list, axis=1)
bol_temp.index = df_bt_ret.index
plt.plot(bol_temp)
plt.xlabel('time', fontsize=12)
plt.ylabel('cumulative return', fontsize=12)
plt.show()
re_bol=np.array(re_bol)
re_bol[:]=6.75-re_bol[:] #6.75 is the best arm(aapl)'s reward

print(re_bol)
print(np.std(re_bol))
print(np.mean(re_bol))
# %%

