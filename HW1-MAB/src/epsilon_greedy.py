#%%
from os.path import join
import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn')
from tools.project_path import *

df_close = pd.read_csv(join(cleaned_data_path, 'US_price.csv'), index_col=0)
df_price = df_close.iloc[:, 0:10]
df_ret = df_price.pct_change()
df_ret = df_ret.dropna()

def epsilon_greedy(exploration_rate):

    n_arm = len(df_price.columns)
    n_operation = np.zeros(n_arm)
    n_operation[:] = len(df_ret[('2016-01-01' <= df_ret.index ) & (df_ret.index <= '2016-01-31')])
    initial_reward_estimate = df_ret[('2016-01-01' <= df_ret.index ) & (df_ret.index <= '2016-01-31')].mean()
    initial_reward_estimate = np.array(initial_reward_estimate)

    df_bt_ret = df_ret[df_ret.index > '2016-01-31']
    mat_bt_ret = np.array(df_bt_ret)

    curr_reward_estimate = initial_reward_estimate
    cum_reward = 1
    best_arm_list = []
    for t in range(len(mat_bt_ret)):
        
        exploration_flag = np.random.binomial(n=1, p=exploration_rate)
        # explore
        if exploration_flag:
            best_arm = np.random.choice(n_arm)
        # exploit
        else:
            best_arm = np.argmax(curr_reward_estimate)

        best_arm_list.append(best_arm)
        best_arm_reward = mat_bt_ret[t, best_arm]
        # update estimate
        curr_reward_estimate[best_arm] = (curr_reward_estimate[best_arm] * n_operation[best_arm] + best_arm_reward) / (n_operation[best_arm] + 1)
        n_operation[best_arm] += 1
        cum_reward *= (best_arm_reward + 1)

    df_weight = pd.DataFrame(index=df_bt_ret.index, columns=df_bt_ret.columns, data=0)
    for t in range(len(df_bt_ret)):
        best_arm = best_arm_list[t]
        df_weight.iloc[t, best_arm] = 1
    df_portfolio_ret = (df_weight * df_bt_ret).sum(axis=1)
    df_pv = np.cumprod(1 + df_portfolio_ret)

    return df_pv, n_operation

df_res = pd.DataFrame(index=df_ret.index)
for i in range(10):
    temp, n_operation = epsilon_greedy(exploration_rate=0.2)
    temp.name = i
    df_res = df_res.join(temp)

plt.plot(df_res.values)
plt.show()
# %%
