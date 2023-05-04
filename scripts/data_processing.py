
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
    ('state', 'TA01_output'),
    ('state', 'TA02_output'),
    ('temperatures', 'DC_GT101_GM101'),
    ('temperatures', 'DC_GT102_GM102'),
    ('temperatures', 'DC_GT103_GM103'),
    ('temperatures', 'DC_GT104_GM104'),
    ('temperatures', 'DC_GT401_GM401'),
    ('temperatures', 'TA01_GT401_GM401'),
    ('temperatures', 'TA02_GT401_GM401'),
    ('temperatures', 'DC_GT301_damped'),
    ('temperatures', 'DC_GT301_outdoor'),
    ('humidity', 'DC_GT101_GM101'),
    ('humidity', 'DC_GT102_GM102'),
    ('humidity', 'DC_GT103_GM103'),
    ('humidity', 'DC_GT104_GM104'),
    ('humidity', 'DC_GT401_GM401'),
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
    ('time', 'minofday_deriv'),
    ('time', 'hourofday')
]

# filter columns to keep only vars
data = data[x_vars].copy()

# remove single NANs
data = data.dropna(how='any')

# remove erroneous setpoints data
data = data[data.setpoints.TA01_GT10X_GM10X != 0.0]

# mash up to minute-based frequency
data = data.groupby(['month', 'day', 'hour', 'minute'], sort=False).mean()

## GH TEMPERATURE
# create "better" estimate of temperature var, w. proper avg.
data[('temperatures', 'TA01_GT10X_GM10X')] = data.temperatures[[
    # ('DC_GT102_GM101'),
    ('DC_GT102_GM102'), 
    ('DC_GT103_GM103'), 
    ('DC_GT104_GM104')
]].values.mean(axis=1)

## DC TEMPERATURE
# min-max scale [btween 0 and 1]
col = ('temperatures', 'DC_GT401_GM401')
data[col] = (data[col] - data[col].min()) / (data[col].max() - data[col].min())

# min-max scale [btween 0 and 1]
col = ('temperatures', 'TA01_GT401_GM401')
data[col] = (data[col] - data[col].min()) / (data[col].max() - data[col].min())

## OUTSIDE TEMP
# min-max scale vals
# set MIN to zero
col = ('temperatures', 'DC_GT301_damped')
data[col] = data[col] + abs(data[col].min())
data[col] = (data[col] - data[col].min()) / (data[col].max() - data[col].min())

## GH HUMIDITY
# create "better" estimate of temperature var, w. proper avg.
data[('humidity', 'TA01_GT10X_GM10X')] = data.humidity[[
    # ('DC_GT102_GM101'),
    ('DC_GT102_GM102'), 
    ('DC_GT103_GM103'), 
    ('DC_GT104_GM104')
]].values.mean(axis=1)

# min-max scale vals
col = ('humidity', 'TA01_GT10X_GM10X')
data[col] = (data[col] - data[col].min()) / (data[col].max() - data[col].min())

## FLOW
col1 = ('state', 'TA01_output')
col2 = ('state', 'TA02_output')
data[col1] = (data[col1] - data[col1].min()) / (data[col1].max() - data[col1].min())
data[col2] = (data[col2] - data[col2].min()) / (data[col2].max() - data[col2].min())

## GSI
# min-max scale GSI
col = ('sun', 'gsi')
data[col] = (data[col] - data[col].min()) / (data[col].max() - data[col].min())

## GSI DERIV
# min-max scale GSI deriv
col = ('sun', 'gsi_deriv')
data[col] = data[col] + abs(data[col].min())
data[col] = (data[col] - data[col].min()) / (data[col].max() - data[col].min())

## WIND
# min-max scale wind vars
col1 = ('wind', 'Wx')
col2 = ('wind', 'Wy')

data[col1] = data[col1] + abs(data[col1].min())
data[col1] = (data[col1] - data[col1].min()) / (data[col1].max() - data[col1].min())
data[col2] = data[col2] + abs(data[col2].min())
data[col2] = (data[col2] - data[col2].min()) / (data[col2].max() - data[col2].min())

## KALMAN FILTERING
cols = [
    (('temperatures', 'TA01_GT10X_GM10X'), 3),
    (('temperatures', 'DC_GT401_GM401'), 2),
    (('temperatures', 'TA01_GT401_GM401'), 3),
    (('temperatures', 'DC_GT301_damped'), 4),
    (('humidity', 'TA01_GT10X_GM10X'), 3),
    # (('sun', 'gsi_deriv'), 3),
    # (('wind', 'Wx'), 3),
    # (('wind', 'Wy'), 3),
    
]

for colSpec in cols:
    # apply KALMAN filter to temperature measurements
    col = colSpec[0]
    varExp = colSpec[-1]
    
    # get data
    X = data[col].values
    n = len(X)

    # process variance, measurement variance
    Q = 1e-5
    R = 0.1**varExp

    xhat=np.zeros(n)      # a posteri estimate of x
    P=np.zeros(n)         # a posteri error estimate
    xhatminus=np.zeros(n) # a priori estimate of x
    Pminus=np.zeros(n)    # a priori error estimate
    K=np.zeros(n)         # gain or blending factor

    # intial guesses
    xhat[0] = X[0]
    P[0] = X[1]

    for k in range(1,n):
        # time update
        xhatminus[k] = xhat[k-1]
        Pminus[k] = P[k-1]+Q

        # measurement update
        K[k] = Pminus[k]/( Pminus[k]+R )
        xhat[k] = xhatminus[k]+K[k]*(X[k]-xhatminus[k])
        P[k] = (1-K[k])*Pminus[k]

    data[col] = xhat

## PREP DATA
t_steps = 10   # 10-min predictions
n_steps = 60   # 60-min backwards look

# Define model variables
model_vars = [
    ('state', 'TA01_output'),
    ('state', 'TA02_output'),
    ('temperatures', 'TA01_GT10X_GM10X'),
    ('temperatures', 'DC_GT401_GM401'),
    ('temperatures', 'TA01_GT401_GM401'),
    ('temperatures', 'DC_GT301_damped'),
    ('humidity', 'TA01_GT10X_GM10X'),
    ('sun', 'gsi'),
    ('sun', 'gsi_deriv'),
    ('time', 'minofday'),
    ('time', 'minofday_deriv'),
    ('setpoints', 'TA01_GT10X_GM10X'),
]

# get data
envData = data[model_vars].copy()

# filter out incomplete days
dayData = envData.groupby(['month', 'day']).count().state
mask = dayData == 1440
dayData = dayData[mask].dropna()
idx = dayData.index.values.tolist()

# filter envData by mask
envData['dayCol'] = envData.index.droplevel(-1).droplevel(-1).values
envData = envData[envData.dayCol.apply(lambda day: day in idx) == True]

del envData['dayCol']

# save data
envData.to_csv(data_path + '\\data_env.csv')
