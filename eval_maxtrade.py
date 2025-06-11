import numpy as np
import pandas as pd


data_dir = './'

lqc_used = np.load(data_dir+'lqc_path.npy')
lasso_bs = np.load(data_dir+'lasso_bootstrap.npy')
cgm_es = np.load(data_dir+'pred_cgm_esloss.npy')
cgm_newloss = np.load(data_dir+'pred_cgm_customloss.npy')

pred_maxindex = np.load(data_dir+'pred_max_index.npy')
pred_minindex = np.load(data_dir+'pred_min_index.npy')

obs = np.load(data_dir+'true_prices.npy')
obs_dayhour = obs[:, :2]
obs_dayhour = obs_dayhour.astype('int')
obs_used = obs[:, 2:]
del obs

## Profit evaluation
obs_used = np.expand_dims(obs_used, axis=1)
# obs_used (4800, 1, 10)
# pred_used (4800, 10000, 10)

lasso_bs = np.moveaxis(lasso_bs, 2, 1)
lqc_used = np.moveaxis(lqc_used, 2, 1)
cgm_es = np.moveaxis(cgm_es, 2, 1)
cgm_newloss = np.moveaxis(cgm_newloss, 2, 1)


# method 3: trajctory based max index
def traj_profit(pred):
    traj_max_index = np.argmax(pred, axis=2)
    
    # Function to find most frequent element along an axis
    def most_frequent(a):
        return np.bincount(a).argmax()
    
    # Find the most frequent element along axis 1
    max_index = np.apply_along_axis(most_frequent, axis=1, arr=traj_max_index)
    
    pred_sell_obs = [obs_used[i, 0, max_index[i]] for i in range(4800)]
    pred_sell_obs = np.array(pred_sell_obs)
    profit = np.sum(pred_sell_obs) / 1000
    print(profit)
    return profit

def traj_cost(pred):
    traj_min_index = np.argmin(pred, axis=2)
    
    # Function to find most frequent element along an axis
    def most_frequent(a):
        return np.bincount(a).argmax()
    
    # Find the most frequent element along axis 1
    min_index = np.apply_along_axis(most_frequent, axis=1, arr=traj_min_index)
    
    pred_buy_obs = [obs_used[i, 0, min_index[i]] for i in range(4800)]
    pred_buy_obs = np.array(pred_buy_obs)
    cost = np.sum(pred_buy_obs) / 1000
    print(cost)
    return cost
    

# Function to find most frequent element along an axis
def most_frequent(a):
    return np.bincount(a).argmax()
    
# only predict index for sell
maxindex_ens = np.argmax(pred_maxindex, axis=1)
# Find the most frequent element along axis 1
max_index = np.apply_along_axis(most_frequent, axis=1, arr=maxindex_ens)

# only predict index for buy
minindex_ens = np.argmax(pred_minindex, axis=1)
# Find the most frequent element along axis 1
min_index = np.apply_along_axis(most_frequent, axis=1, arr=minindex_ens)


profit_cost = np.zeros((5, 2))

profit_cost[0, 0] = traj_profit(lasso_bs)
profit_cost[1, 0] = traj_profit(lqc_used)
profit_cost[2, 0] = traj_profit(cgm_es)
profit_cost[3, 0] = traj_profit(cgm_newloss)

pred_sell_obs = [obs_used[i, 0, max_index[i]] for i in range(4800)]
pred_sell_obs = np.array(pred_sell_obs)
profit_cost[4, 0] = np.sum(pred_sell_obs) / 1000

profit_cost[0, 1] = traj_cost(lasso_bs)
profit_cost[1, 1] = traj_cost(lqc_used)
profit_cost[2, 1] = traj_cost(cgm_es)
profit_cost[3, 1] = traj_cost(cgm_newloss)

pred_buy_obs = [obs_used[i, 0, min_index[i]] for i in range(4800)]
pred_buy_obs = np.array(pred_buy_obs)
profit_cost[4, 1] = np.sum(pred_buy_obs) / 1000


print('lasso_bs, lqc, cgm es, cgm loss, nn index | sell, buy')
print(profit_cost[:, 0])
print(profit_cost[:, 1])
np.save(data_dir+'results/profit_cost.npy', profit_cost)
