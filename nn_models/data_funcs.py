
import os
import pandas as pd
import numpy as np


# define function for data frequency to M minutes
def data_reduce(data, m):
    idxObj = zip(
        data.index.get_level_values(0),
        data.index.get_level_values(1),
        data.index.get_level_values(2),
        data.index.get_level_values(3) // m
    )

    index = pd.MultiIndex.from_tuples(
        tuples=idxObj,
        names=['month', 'day', 'hour', 'minute']
    )

    data.index = index
    dates = data.groupby(['month', 'day', 'hour', 'minute'], sort=False).last()[('time', 'date')]
    data = data.groupby(['month', 'day', 'hour', 'minute'], sort=False).mean()
    return data, dates

# define function f. flagging erroneous sequences
def date_flagger(data, n_steps):
    # create flag for erroneous sequences
    data['hour'] = data.index.get_level_values(2).values    
    data['date_flag'] = data.hour - data.hour.shift(n_steps) > 1
    
    # get positions in data, w.r.t. n_step removed observations at start
    flagged_idx = np.where(data.date_flag.values == 1)
    flagged_idx = flagged_idx[0] - n_steps
    
    del data['hour'], data['date_flag']
    
    return flagged_idx

# define func f. creating sequences
def seq_maker(data, targets, temps, dates, t_steps, n_steps):
    
    vals = data.values
    sequences = []
    for i in range(len(vals) - n_steps):
        sequences.append(vals[i:i+n_steps])
    sequences = np.stack(sequences)
    
    flags = date_flagger(data, n_steps)
    mask = [idx not in flags for idx in range(len(sequences))]
    
    sequences = sequences[mask]
    targets = targets[n_steps:][mask].values[:, np.newaxis]
    temps_t = temps[n_steps-t_steps:-t_steps][mask].values[:, np.newaxis]
    temps = temps[:-n_steps][mask].values[:, np.newaxis]
    dates = dates[:-n_steps][mask].values
    
    return sequences, targets, temps, temps_t, dates

# define func f. data normalization
def data_norm(data_train, data_test, data_val):
    col_params = {}
    for col in data_train.columns:

        min_val = data_train[col].min()
        max_val = data_train[col].max()

        # normalize
        mean = data_train[col].mean()
        std = data_train[col].std()

        data_train[col] = (data_train[col] - mean) / std
        data_test[col] = (data_test[col] - mean) / std
        data_val[col] = (data_val[col] - mean) / std

        col_params[col] = {
            'mean':mean,
            'std':std,
            'max':max_val,
            'min':min_val
            }
    
    return data_train, data_test, data_val, col_params

def k_fold_data(data, k_idx, k_frac, m, cols, t_steps, n_steps, setpoint, shuffle):
    
    # get days
    days = data.groupby(['month', 'day'], sort=False).count().index.values
    
    # get days for K:th fold
    # train_n = int(len(days) * (1 - k_frac))
    # test_n = len(days) - train_n
    test_n = int(len(days) * k_frac)
    
    # split days by test and train
    days_test = days[int(k_idx*test_n):int((k_idx+1)*test_n)].tolist()
    mask_test = np.array([day in days_test for day in data.index.droplevel(-1).droplevel(-1).droplevel(-1).values])
    data_train = data.loc[~mask_test].copy()
    data_test = data.loc[mask_test].copy()

    # get validation data
    days_train = np.unique(data_train.index.droplevel(-1).droplevel(-1).droplevel(-1).values)
    val_n = int(len(days_train) * k_frac)
    days_val = np.random.choice(days_train, val_n, replace=False).tolist()
    mask_val = np.array([day in days_val for day in data_train.index.droplevel(-1).droplevel(-1).droplevel(-1).values])
    data_val = data_train.loc[mask_val].copy()
    data_train = data_train.loc[~mask_val].copy()
    
    # reduce to m-min observations
    data_train, dates_train = data_reduce(data_train, m)
    data_test, dates_test = data_reduce(data_test, m)
    data_val, dates_val = data_reduce(data_val, m)

    # remove setpoint
    if setpoint:
        data_train[('temperatures', 'TA01_GT10X_GM10X')] += 20 - data_train.setpoints.TA01_GT10X_GM10X
        data_test[('temperatures', 'TA01_GT10X_GM10X')] += 20 - data_test.setpoints.TA01_GT10X_GM10X
        data_val[('temperatures', 'TA01_GT10X_GM10X')] += 20 - data_val.setpoints.TA01_GT10X_GM10X
    
    # filter data
    data_train = data_train[cols].copy()
    data_test = data_test[cols].copy()
    data_val = data_val[cols].copy()
    
    # normalize
    data_train, data_test, data_val, col_params = data_norm(data_train, data_test, data_val)
    
    # get targets
    targets_train = data_train.pop(('temperatures', 'TA01_GT10X_GM10X'))
    targets_test = data_test.pop(('temperatures', 'TA01_GT10X_GM10X'))
    targets_val = data_val.pop(('temperatures', 'TA01_GT10X_GM10X'))
    
    # get temp info
    temps_train = targets_train.copy()
    temps_test = targets_test.copy()
    temps_val = targets_val.copy()
    
    # create sequences    
    sequences_train, targets_train, temps_train, temps_t_train, dates_train = seq_maker(data_train, targets_train, temps_train, dates_train, t_steps, n_steps)
    sequences_test, targets_test, temps_test, temps_t_test, dates_test = seq_maker(data_test, targets_test, temps_test, dates_test, t_steps, n_steps)
    sequences_val, targets_val, temps_val, temps_t_val, dates_val = seq_maker(data_val, targets_val, temps_val, dates_val, t_steps, n_steps)
    
    # create MASKED sequences
    sequences_masked = sequences_test.copy()
    for t in range(1, t_steps):
        sequences_masked[:, -t, :] = sequences_masked[:, -(t_steps), :]

    # shuffle training data randomly
    if shuffle:
        idxs = np.arange(len(targets_train))
        np.random.shuffle(idxs)

        sequences_train = sequences_train[idxs]
        targets_train = targets_train[idxs]
        temps_train = temps_train[idxs]
        temps_t_train = temps_t_train[idxs]
        dates_train = dates_train[idxs]
    
    # return tups w. train and test
    train_tup = (sequences_train, targets_train, temps_train, temps_t_train, dates_train)
    test_tup = (sequences_test, targets_test, temps_test, temps_t_test, sequences_masked, dates_test)
    val_tup = (sequences_val, targets_val, temps_val, temps_t_val, dates_val)
    
    return train_tup, test_tup, val_tup, col_params