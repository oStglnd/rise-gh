
import os
import pandas as pd

# path for RISE data
home_path = os.getcwd()
data_path = home_path + '\\data\\sensors\\'
save_path = home_path + '\\data\\control_v2.csv'

# data categories
data_cat = [
   'flow',
   'state'
] 

names = [
    'DC_SP103', 
    'DC_SP104', 
    'DC_SP105', 
    'DC_SP106', 
    'DC_SP107',
    'DC_SP108', 
    'DC_SP110', 
    'DC_SP111', 
    'DC_SP112', 
    'DC_SP113',
    'FF01_GP101', 
    'FF01_output', 
    'FF02_GP101', 
    'FF02_output',
    'TA01_GP101', 
    'TA01_SP101', 
    'TA01_SP101_output', 
    'TA01_SP102',
    'TA01_SP102_output', 
    'TA01_output', 
    'TA02_GP101', 
    'TA02_SP109',
    'TA02_SP109_output', 
    'TA02_output'
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
            data[(cat, name, typ)] = dataset[dtype].values

dates = dataset.Timestamps.apply(
    lambda date: pd.to_datetime(date, unit='s')
)

data = pd.DataFrame(
    index=dates,
    data=data
)

data = data.reindex(
    sorted(data.columns,
           key=lambda c: c[0]),
    axis=1
)

# REMOVE NANS (same as for CLIMATE data
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