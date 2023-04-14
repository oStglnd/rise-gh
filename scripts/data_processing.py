
import os
import pandas as pd
import numpy as np

# define paths
home_path = os.path.dirname(os.getcwd())
data_path = home_path + '\\data\\'

# get merged data
data = pd.read_csv(
    data_path + 'data_merged.csv',
    header=[0, 1],
    index_col=[0, 1, 2, 3, 4]
)

# define vars
x_vars = [
    ('flow', 'TA01_GP101'),
    ('flow', 'FF01_GP101'),
    ('temperatures', 'DC_GT401_GM401'),
    ('temperatures', 'TA01_GT10X_GM10X'),
    ('temperatures', 'DC_GT301_damped'),
    ('temperatures', 'DC_GT301_outdoor'),
    ('setpoints', 'TA01_GT10X_GM10X'),
    ('humidity', 'TA01_GT10X_GM10X'),
    ('humidity', 'DC_GT401_GM401'),
    ('sun', 'gsi'),
    ('sun', 'vol'),
    ('time', 'mod'),
    ('time', 'doy')
]

# filter columns to keep only vars
data = data[x_vars].copy()

# remove single NANs
data = data.dropna(how='any')

# remove erroneous setpoints data
data = data[data.setpoints.TA01_GT10X_GM10X != 0.0]

# Transform setpoints variable to instead account for difference w.r.t 20 deg C
data[('temperatures', 'setpoint_diff')] = data.setpoints.TA01_GT10X_GM10X - 20.0
del data[('setpoints', 'TA01_GT10X_GM10X')]

# remove "OUTLIERS" from DC-TEMP
data[('temperatures', 'DC_GT401_GM401_roll')] = data.temperatures.DC_GT401_GM401.rolling(window=240, center=False).mean()
data[('temperatures', 'DC_diff')] = np.abs(data.temperatures.DC_GT401_GM401 - data.temperatures.DC_GT401_GM401_roll)
data.loc[data.temperatures.DC_diff > 2, ('temperatures', 'DC_GT401_GM401')] = data.temperatures.DC_GT401_GM401_roll

# remove "OUTLIERS" from DC-FLOW
data[('flow', 'TA01_GP101')] = data.flow.TA01_GP101.apply(lambda val: max(1800, val))

# make flow OUT negative
data[('flow', 'FF01_GP101')] = - data.flow.FF01_GP101

# get FFT-smoothed series
def fft_smoothing(data, spacing, threshold):
    fourier = np.fft.rfft(data.values)
    freqs = np.fft.rfftfreq(
        n = len(data),
        d = spacing
    )
    
    fourier[freqs > threshold] = 0
    filtered = np.fft.irfft(fourier)
    return filtered

data.loc[:, ('temperatures', 'DC_GT401_GM401_fft')] = fft_smoothing(
    data=data[('temperatures', 'DC_GT401_GM401')],
    spacing=1/100,
    threshold=1
)

data.loc[:, ('temperatures', 'TA01_GT10X_GM10X_fft')] = fft_smoothing(
    data=data[('temperatures', 'TA01_GT10X_GM10X')],
    spacing=1/10,
    threshold=1/10
)

data.loc[:, ('humidity', 'DC_GT401_GM401_fft')] = fft_smoothing(
    data=data[('humidity', 'DC_GT401_GM401')],
    spacing=1/10,
    threshold=1/10
)

data[('humidity', 'DC_GT401_GM401_fft')] = data.humidity.DC_GT401_GM401_fft.apply(
    lambda val: max(0, val)
)

### REMOVE "INCOMPLETE" days
idxs = data.groupby(['month', 'day']).count().flow.TA01_GP101 == 2880
idxs = idxs[idxs == 1]
data = data[idxs]

# save data
data.to_csv(data_path + '\\data_env.csv')
