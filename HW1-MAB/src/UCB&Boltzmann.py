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

    return  cum_reward_series,n_operation


def boltzmann(n_arm,initial_reward_estimate,mat_bt_ret, sigma):
    n_operation = np.zeros(n_arm)
    curr_reward_estimate = initial_reward_estimate
    cum_reward=1
    
    for t in range(len(mat_bt_ret)):
        
        # probability of arm to be selected based on the current estimated reward and Boltzmann distribution
        reward_prob = np.exp(np.array(curr_reward_estimate)/sigma)/np.exp(np.array(curr_reward_estimate)/sigma).sum()
        
        # select arm based on previous probability distrivution
        best_arm = np.random.choice(n_arm, size=1, p=reward_prob)[0]
        best_arm_reward = mat_bt_ret[t,best_arm]
  
        curr_reward_estimate[best_arm] = (curr_reward_estimate[best_arm] * n_operation[best_arm] + best_arm_reward)/(n_operation[best_arm] + 1)
        n_operation[best_arm] += 1
        cum_reward*=(1+ best_arm_reward)  

    return cum_reward, curr_reward_estimate, n_operation   
#%%
#run simulatioin
N=10  #number of path
c_grad=np.arange(-4,4.1,0.5)
ubc_temp=pd.DataFrame(np.zeros(len(mat_bt_ret)))
'''
#select best c for UCB
for i in c_grad:
    temp_result=0
    for j in range(N):
         temp_result+=(ucb(mat_bt_ret, i, n_arm)[0])
    ubc_result.append(temp_result/N)

plt.plot(c_grad,ubc_result, c='firebrick')
plt.xlabel('c', fontsize=12)
plt.ylabel('total return', fontsize=12)
plt.xlim(0, 1)
plt.show()

best_c=c_grad[np.argmax(ubc_result)]
'''

temp_list = []
#df_res = pd.DataFrame(index=df_ret.index)
for i in range(N):
    temp, n_operation = ucb(mat_bt_ret, 2, n_arm)
    temp = pd.DataFrame(temp)
    temp.columns = [i]
    temp_list.append(temp)

ubc_temp = pd.concat(temp_list, axis=1)
ubc_temp.index = df_bt_ret.index
plt.plot(ubc_temp)
plt.xlabel('time', fontsize=12)
plt.ylabel('cumulative return', fontsize=12)
plt.show()
#%%
#SIGMA OF boltzmann
sigma_grad=np.arange(0.01,1.01,0.2)
for i in sigma_grad:
    temp_result=0
    for j in range(N):
         temp_result+=(boltzmann(n_arm, initial_reward_estimate,mat_bt_ret, i)[0])
    sigma_result.append(temp_result/N)

plt.plot(sigma_grad,sigma_result, c='firebrick')
plt.xlabel('sigma', fontsize=12)
plt.ylabel('total return', fontsize=12)
plt.xlim(0, 1)
plt.show()





''''
df_res = pd.DataFrame(index=df_ret.index)
for i in range(10):
    temp, n_operation = epsilon_greedy(exploration_rate=0.1
    temp.name = i
    df_res = df_res.join(temp)

plt.plot(df_res.dropna())
plt.show()
'''
# %%

