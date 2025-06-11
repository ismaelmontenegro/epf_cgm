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


# Initialize SCP values
P = np.arange(95, 0, -5)
profits_scp = np.zeros((len(P), 5))
profits_scp[:, 0] = P
costs_scp = profits_scp.copy()


def scp_upper(pred_used, n_ens=10000):
    # Initialize result array
    res_upper = np.zeros((4800, len(P), 10))
    profits_arr = np.zeros((len(P)))
    
    for index in range(4800):
        real_traj = obs_used[index, :, :] # (1, 10)
        curr_traj = pred_used[index, :, :] # (n_ens, 10)
        no_traj = n_ens
        idxs = range(n_ens)
        
        for p in range(len(P)):
            while curr_traj.shape[0] > 0.01 * P[p] * no_traj:
                idxs = np.arange(curr_traj.shape[0])
                
                I1 = np.argmax(curr_traj, axis=0)
                # indexes of trajectories with maximum prices
                
                for q in range(10):
                    idxs = idxs[idxs != I1[q]]  # removing these indices
                curr_traj = curr_traj[idxs, :]  # selecting only remaining indices
                
            # upper prediction band is set of maximum prices in each timepoint
            upper = np.max(curr_traj, axis=0)
            
            # trading -> using prediction band and real trajectory
            res_upper[index, p, :] = upper
    
    for k in range(len(P)):
        pred_upper = res_upper[:, k, :]
        pred_index = np.argmax(pred_upper, axis=1)
        pred_sell_obs = [obs_used[i, 0, pred_index[i]] for i in range(4800)]
        pred_sell_obs = np.array(pred_sell_obs)
        profit_get = np.sum(pred_sell_obs) / 1000
        profits_arr[k] = profit_get

    return profits_arr


def scp_lower(pred_used, n_ens=10000):
    res_lower = np.zeros((4800, len(P), 10))
    profits_arr = np.zeros((len(P)))
    
    for index in range(4800):
        real_traj = obs_used[index, :, :] # (1, 10)
        curr_traj = pred_used[index, :, :] # (n_ens, 10)
        no_traj = n_ens
        idxs = range(n_ens)
        
        for p in range(len(P)):
            while curr_traj.shape[0] > 0.01 * P[p] * no_traj:
                idxs = np.arange(curr_traj.shape[0])
                
                I1 = np.argmin(curr_traj, axis=0)
                
                for q in range(10):
                    idxs = idxs[idxs != I1[q]]  # removing these indices
                curr_traj = curr_traj[idxs, :]  # selecting only remaining indices
                
            lower = np.min(curr_traj, axis=0)
            
            res_lower[index, p, :] = lower
    
    for k in range(len(P)):
        pred_lower = res_lower[:, k, :]
        pred_index = np.argmax(pred_lower, axis=1)
        pred_sell_obs = [obs_used[i, 0, pred_index[i]] for i in range(4800)]
        pred_sell_obs = np.array(pred_sell_obs)
        profit_get = np.sum(pred_sell_obs) / 1000
        profits_arr[k] = profit_get

    return profits_arr

profits_scp[:, 1] = scp_upper(lqc_used)
profits_scp[:, 2] = scp_upper(lasso_bs)
profits_scp[:, 3] = scp_upper(cgm_es)
profits_scp[:, 4] = scp_upper(cgm_newloss)
np.save(data_dir+'scp_profits.npy', profits_scp)

costs_scp[:, 1] = scp_lower(lqc_used)
costs_scp[:, 2] = scp_lower(lasso_bs)
costs_scp[:, 3] = scp_lower(cgm_es)
costs_scp[:, 4] = scp_lower(cgm_newloss)
np.save(data_dir+'scp_costs.npy', costs_scp)

print('sell upper band: profits_scp')
print(profits_scp)
print('sell lower band: costs_scp')
print(costs_scp)
