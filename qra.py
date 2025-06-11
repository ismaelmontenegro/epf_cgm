import numpy as np
import pandas as pd
import time
from sklearn.linear_model import QuantileRegressor
from multiprocessing import Pool


home_dir = '/hkfs/home/haicore/stat/gm2154/project3_epf/data/'
data = pd.read_feather(home_dir+'lasso_xy.feather')

arr = data.values
arr_re = np.reshape(arr, (10, 440, 24, 5))
# axis 0: 10 -> 1
# axis 1: 397 -> 837
# axis 3: 0 -> 23
# axis 4: day, hour, traj, pred, obs

qra_arr = np.zeros((10, 320, 24, 100))

q_seq = np.arange(0.01, 1.0, 0.01)

start = time.time()

def process_day(args):
    traj, hour, day = args
    train_x = arr_re[traj, day:(day+120), hour, 3]
    train_y = arr_re[traj, day:(day+120), hour, 4]
    test_x = arr_re[traj, (day+120), hour, 3]
    test_y = arr_re[traj, (day+120), hour, 4]
    
    train_x = np.expand_dims(train_x, axis=1)
    test_x = np.expand_dims(test_x, axis=[0,1])
    
    results = np.zeros(100)
    results[99] = test_y
    
    for i in range(99):
        q = q_seq[i]
        reg = QuantileRegressor(quantile=q, solver='highs').fit(train_x, train_y)
        q_pred = reg.predict(test_x)
        results[i] = q_pred[0]
    
    return traj, day, hour, results

def update_qra_arr(result):
    traj, day, hour, results = result
    qra_arr[traj, day, hour, :] = results

with Pool() as pool:
    for traj in range(10):
        print('traj', traj)
        for hour in range(24):
            print('hour', hour)
            tasks = [(traj, hour, day) for day in range(320)]
            for result in pool.imap_unordered(process_day, tasks):
                update_qra_arr(result)
                
                
#for traj in range(10):
#    print('traj', traj)
#    for hour in range(24):
#        print('hour', hour)
#        for day in range(320):
#            print('day', day+120+397+1)
#            train_x = arr_re[traj, day:(day+120), hour, 3]
#            train_y = arr_re[traj, day:(day+120), hour, 4]
#            test_x = arr_re[traj, (day+120), hour, 3]
#            test_y = arr_re[traj, (day+120), hour, 4]
#            
#            train_x = np.expand_dims(train_x, axis=1)
#            test_x = np.expand_dims(test_x, axis=[0,1])
#            
#            qra_arr[traj, day, hour, 99] = test_y
#            
#            for i in range(99):
#                q = q_seq[i]
#                reg = QuantileRegressor(quantile=q, solver='highs').fit(train_x, train_y)
#                q_pred = reg.predict(test_x)
#                
#                qra_arr[traj, day, hour, i] = q_pred[0]
                
duration = time.time() - start
print(duration/60)

np.save(home_dir+'qra_lasso.npy', qra_arr)
