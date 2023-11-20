
# import dependencies
import os
import json
import pandas as pd
import numpy as np

# define relevant paths
home_path = os.path.dirname(os.getcwd())
data_path = home_path + '\\data\\'
plot_path = home_path + '\\plotting\\plots\\'
save_path = home_path + '\\model\\saved\\'

# get merged data
data = pd.read_csv(
    data_path + 'data_merged.csv',
    header=[0, 1],
    index_col=[0, 1, 2, 3, 4]
)

# convert index.date col to datetime
#data.index = pd.to_datetime(data.index.values)
data.loc[:, ('time', 'date')] = pd.to_datetime(data.time.date)

# define X vars
x_vars = [
    ('flow', 'TA01_GP101'),
    ('flow', 'TA02_GP101'),
    ('state', 'TA01_output'),
    ('state', 'TA02_output'),
    ('power', 'phase'),
    ('temperatures', 'TA01_GT10X_GM10X'),
    ('temperatures', 'DC_GT101_GM101'),
    ('temperatures', 'DC_GT102_GM102'),
    ('temperatures', 'DC_GT103_GM103'),
    ('temperatures', 'DC_GT104_GM104'),
    ('temperatures', 'DC_GT401_GM401'),
    ('temperatures', 'TA01_GT401_GM401'),
    ('temperatures', 'TA02_GT401_GM401'),
    ('temperatures', 'DC_GT301_damped'),
    ('temperatures', 'DC_GT301_outdoor'),
    ('humidity', 'TA01_GT10X_GM10X'),
    ('humidity', 'DC_GT101_GM101'),
    ('humidity', 'DC_GT102_GM102'),
    ('humidity', 'DC_GT103_GM103'),
    ('humidity', 'DC_GT104_GM104'),
    ('humidity', 'DC_GT401_GM401'),
    ('humidity', 'TA01_GT401_GM401'),
    ('humidity', 'TA02_GT401_GM401'),
    ('humidity', 'outdoor'),
    ('setpoints', 'TA01_GT10X_GM10X'),
    ('sun', 'gsi'),
    ('sun', 'gsi_deriv'),
    ('sun', 'vol'),
    ('sun', 'vol_deriv'),
    ('wind', 'Wx'),
    ('wind', 'Wy'),
    ('time', 'dayofyear'),
    ('time', 'monthofyear'),
    ('time', 'minofday'),
    ('time', 'hourofday'),
    ('time', 'date')
]

# filter columns to keep only x_vars
data = data[x_vars].copy()

# remove single NAN
data = data.dropna(how='any')

# remove presumably erroneous measurements
cols = [
    'DC_GT101_GM101',
    'DC_GT102_GM102',
    'DC_GT103_GM103',
    'DC_GT104_GM104'
]

for col in cols:
    data[('flag', col)] = (((data.temperatures[col] - data.temperatures[col].shift(1)) < -1) \
                            & ((data.humidity[col] - data.humidity[col].shift(1)) > 5)) \
                            | (data.humidity[col] > 70)

    hrs = data.groupby(['month', 'day', 'hour']).sum()[('flag', col)] > 0
    hrsDrop = hrs[hrs == 1].index.values
    mask = np.array([hr not in list(hrsDrop) for hr in data.index.droplevel(-1).droplevel(-1).values])
    data = data[mask]
    
    hrs = data.groupby(['month', 'day', 'hour']).var()[('humidity', col)] > 40
    hrsDrop = hrs[hrs == 1].index.values
    mask = np.array([hr not in list(hrsDrop) for hr in data.index.droplevel(-1).droplevel(-1).values])
    data = data[mask]


# process humidity data
def abs_humid(temp, rel_humid):
    abs_humidity =  6.112 * np.exp(17.67 * temp / (temp + 243.5)) * rel_humid * 2.1674 / (273.15 + temp)
    return abs_humidity

def rel_humid(temp, abs_humid):
    rel_humidity = abs_humid * (273.15 + temp) / (6.112 * np.exp(17.67 * temp / (temp + 243.5)) * 2.1674)
    return rel_humidity

# define sensors to calculate abs. humidity for
sensors = [
    'DC_GT101_GM101',
    'DC_GT102_GM102',
    'DC_GT103_GM103',
    'DC_GT104_GM104',
    'TA01_GT10X_GM10X',
    'TA01_GT401_GM401',
    'TA02_GT401_GM401'
]

for sensor in sensors:
    data[('humidity', sensor + '_abs')] = abs_humid(data.temperatures[sensor], data.humidity[sensor])

# data[('humidity', 'outdoor_abs')] = abs_humid(data.temperatures.DC_GT301_outdoor, data.humidity.outdoor)
data[('humidity', 'outdoor_abs')] = abs_humid(data.temperatures.DC_GT301_damped, data.humidity.outdoor)

# get humidity difference w.r.t. GH
data[('humidity', 'TA01_GT401_GM401_rel')] = data[('humidity', 'TA01_GT401_GM401_abs')] - data[('humidity', 'TA01_GT10X_GM10X_abs')]
data[('humidity', 'TA02_GT401_GM401_rel')] = data[('humidity', 'TA02_GT401_GM401_abs')] - data[('humidity', 'TA01_GT10X_GM10X_abs')]
data[('humidity', 'outdoor_rel')] = data[('humidity', 'outdoor_abs')] - data[('humidity', 'TA01_GT10X_GM10X_abs')]

data[('temperatures', 'TA01_GT401_GM401_rel')] = data[('temperatures', 'TA01_GT401_GM401')] - data[('temperatures', 'TA01_GT10X_GM10X')]
data[('temperatures', 'TA02_GT401_GM401_rel')] = data[('temperatures', 'TA02_GT401_GM401')] - data[('temperatures', 'TA01_GT10X_GM10X')]
data[('temperatures', 'DC_GT301_outdoor_rel')] = data[('temperatures', 'DC_GT301_outdoor')] - data[('temperatures', 'TA01_GT10X_GM10X')]
data[('temperatures', 'DC_GT301_damped_rel')] = data[('temperatures', 'DC_GT301_damped')] - data[('temperatures', 'TA01_GT10X_GM10X')]

# get SCALED temp and humid
ta01_min = 35
ta01_max = 75

ta02_min = 0
ta02_max = 85

# get outputs relative to TA01_min
data[('state', 'TA01_output_minmax')] = data.state.TA01_output / ta01_min
data[('state', 'TA02_output_minmax')] = data.state.TA02_output / ta01_min

# get scaled temperature inflow
data[('temperatures', 'TA01_GT401_GM401_scaled')] = data.temperatures.TA01_GT401_GM401_rel * data.state.TA01_output_minmax
data[('temperatures', 'TA02_GT401_GM401_scaled')] = data.temperatures.TA02_GT401_GM401_rel * data.state.TA02_output_minmax
data[('temperatures', 'DC_GT301_damped_scaled')] = data.temperatures.DC_GT301_damped_rel * data.state.TA02_output_minmax
data[('temperatures', 'DC_GT301_outdoor_scaled')] = data.temperatures.DC_GT301_outdoor_rel * data.state.TA02_output_minmax
data[('temperatures', 'TA_inflow')] = data.temperatures.TA01_GT401_GM401_scaled + data.temperatures.TA02_GT401_GM401_scaled
data[('temperatures', 'TA_inflow_out')] = data.temperatures.TA01_GT401_GM401_scaled + data.temperatures.DC_GT301_damped_scaled


# get scaled humidity inflow
data[('humidity', 'TA01_GT401_GM401_scaled')] = data.humidity.TA01_GT401_GM401_rel * data.state.TA01_output_minmax
data[('humidity', 'TA02_GT401_GM401_scaled')] = data.humidity.TA02_GT401_GM401_rel * data.state.TA02_output_minmax
data[('humidity', 'outdoor_scaled')] = data.humidity.outdoor_rel * data.state.TA02_output_minmax
data[('humidity', 'TA_inflow')] = data.humidity.TA01_GT401_GM401_scaled + data.humidity.TA02_GT401_GM401_scaled
data[('humidity', 'TA_inflow_out')] = data.humidity.TA01_GT401_GM401_scaled + data.humidity.outdoor_scaled

### create "better" estimate of temperature var, w. proper avg.
data[('temperatures', 'TA01_GT10X_GM10X')] = data.temperatures[[
#     ('DC_GT101_GM101'),
    ('DC_GT102_GM102'), 
    ('DC_GT103_GM103'), 
    ('DC_GT104_GM104')
]].values.mean(axis=1)

### create "better" estimate of humidity var, w. proper avg.
data[('humidity', 'TA01_GT10X_GM10X')] = data.humidity[[
    # ('DC_GT102_GM101'),
    ('DC_GT102_GM102'), 
    ('DC_GT103_GM103'), 
    ('DC_GT104_GM104')
]].values.mean(axis=1)

### create "better" estimate of absolute humidity var, w. proper avg.
data[('humidity', 'TA01_GT10X_GM10X_abs')] = data.humidity[[
    # ('DC_GT102_GM101_abs'),
    ('DC_GT102_GM102_abs'), 
    ('DC_GT103_GM103_abs'), 
    ('DC_GT104_GM104_abs')
]].values.mean(axis=1)


### get thermal loss input
data[('temperatures', 'TA01_GT10X_GM10X_loss')] = data.temperatures.TA01_GT10X_GM10X - data.temperatures.DC_GT301_damped

# save as processed data
data.to_csv(data_path + 'data_processed.csv')
