import numpy as np
from scipy.stats import norm
from scipy.special import erfinv
from multiprocessing import Pool
import time

home_dir = '/hkfs/home/haicore/stat/gm2154/project3_epf/data/'

qra_arr = np.load(home_dir+'qra_lasso.npy')
# (10, 320, 24, 100)
# reshape from lass_xy
# arr_re = np.reshape(lass_xy, (10, 440, 24, 5))
# axis 0: 10 -> 1
# axis 1: days
# axis 3: 0 -> 23
# axis 4: quantiles + real

cal_len = 320
Y = np.zeros((320, 24, 10))

qra_arr_used = np.moveaxis(qra_arr, 0, 2) # (320, 24, 10, 100)
real = qra_arr_used[:, :, :, 99]
qra_99q = qra_arr_used[:, :, :, :99]
QRA = np.sort(qra_99q, axis=-1)

for day in range(cal_len):
    for hour in range(24):
        for traj in range(10):
            real_values = real[day, hour, traj]
            qra_values = QRA[day, hour, traj, :]
            
            greater_count = np.sum(real_values > qra_values)
            
            if greater_count / 99 == 0:
                Y[day, hour, traj] = 0.01
            elif greater_count / 99 == 1:
                Y[day, hour, traj] = 0.99
            else:
                Y[day, hour, traj] = greater_count / 99


def calc_quants(tr, quantiles):
    # tr (10000,); quantiles (101,)
    quants = np.zeros(tr.shape)
    for i in range(len(tr)):
        idxs = tr[i] > np.arange(101)
        idx = np.max(np.where(idxs)[0])
        up = quantiles[idx + 1]
        down = quantiles[idx]
        quants[i] = down + (up - down) * (tr[i] - idx)
    return quants


# Define the parallel worker function
def process_trajectory(i, real, QRA, sigma_cal, n_ens, X):
    traj = np.zeros((24, 10, n_ens))
    
    for h in range(24):
        maxes = np.max(real[(i-sigma_cal):i, h, :], axis=0)
        mins = np.min(real[(i-sigma_cal):i, h, :], axis=0)
        C = np.corrcoef(X[(i-sigma_cal):i, h, :], rowvar=False)
        samples = np.random.multivariate_normal(np.zeros(10), C, n_ens)
        tr = norm.cdf(samples)
        tr = tr * 100
        
        for quar in range(10):
            quantiles = QRA[i, h, quar, :]
            min0 = np.min([quantiles[0], mins[quar]])
            max0 = np.max([quantiles[98], maxes[quar]])
            # approximation of quantiles at the edges
            quantiles_edge0 = np.append(min0, quantiles)
            quantiles_edge = np.append(quantiles_edge0, max0)
            
            traj[h, quar, :] = calc_quants(tr[:, quar], quantiles_edge)
            
    return traj



cal_len = 320
sigma_cal = 120
n_ens = 10000

X = np.sqrt(2) * erfinv(2 * Y - 1)

real_path = real[sigma_cal:, :, :] # (200, 24, 10)
trajectories = np.zeros((cal_len-sigma_cal, 24, 10, n_ens))
# (200, 24, 10, 10000)


start = time.time()
# Parallel processing using Pool
with Pool() as pool:
    results = pool.starmap(process_trajectory, [(i, real, QRA, sigma_cal, n_ens, X) for i in range(sigma_cal, cal_len)])

end = time.time()


for i, traj in enumerate(results):
    trajectories[i, :, :, :] = traj


real_path0 = np.reshape(real_path, (4800, 10))

trajectories0 = np.reshape(trajectories, (4800, 10, n_ens))
trajectories0 = trajectories0.astype(np.float32)

np.save(home_dir+'real_path.npy', real_path0)
np.save(home_dir+'lqc_path.npy', trajectories0)

duration = end - start
print(real_path.shape)
print(trajectories.shape)
print('Total Computation Time:', duration)


