
import os
import pandas as pd

# path for RISE data
home_path = os.getcwd()
data_path = home_path + '\\data\\sensors\\'
save_path = home_path + '\\data\\climate_sensors_v2.csv'

# data categories
data_cat = [
   'humidity',
   'temperatures',
   'pressure',
] 

names = [
    'DC_GP101',
    'DC_GT101_GM101',
    'DC_GT102_GM102',
    'DC_GT103_GM103',
    'DC_GT104_GM104',
    'DC_GT301_damped',
    'DC_GT301_outdoor',
    'DC_GT401_GM401',
    'FF01_GP101',
    'FF02_GP101',
    'TA01_GP101',
    'TA01_GP102',
    'TA01_GP103',
    'TA01_GP401',
    'TA01_GT10X_GM10X',
    'TA01_GT401_GM401',
    'TA02_GP101',
    'TA02_GP401',
    'TA02_GT401_GM401'
]

data = {}
for cat in data_cat:
    file = data_path + cat + '.csv'
    dataset = pd.read_csv(file)
    
    cols = dataset.columns.values.tolist()

    for name in names:
        dtypes = [col for col in cols if name in col]
        
        for dtype in dtypes:
            typ = dtype.split('__')[-1]
            data[(cat, name)] = dataset[dtype].values

dates = dataset.Timestamps.apply(
    lambda date: pd.to_datetime(date, unit='s')
).rename('date')

data = pd.DataFrame(
    index=dates,
    data=data
)

data = data.reindex(
    sorted(data.columns,
            key=lambda c: c[0]),
    axis=1
)

# REMOVE NANS
drop_dates = [
    '2022-12-12',
    '2022-12-21',
    '2022-12-24',
    '2022-12-25',
    '2022-12-26',
    '2022-12-27',
    '2022-12-28',
    '2023-01-27',
    '2023-02-13',
]
data['date'] = data.index
data = data[data.date.apply(lambda d: d.date().isoformat() not in drop_dates)]
del data['date']

data.to_csv(save_path)