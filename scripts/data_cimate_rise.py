
import os
import pandas as pd
import json

# path for RISE data
home_path = os.path.dirname(os.getcwd())
data_path = home_path + '\\data\\sensors\\'
save_path = home_path + '\\data\\climate_sensors_v2.csv'

# get data specs
specs = json.load(open('misc\\specs.json'))
names = specs['control']
drop_dates = specs['dates']

# data categories
data_cat = [
   'humidity',
   'temperatures',
   'pressure',
]

# iterate over data categories, add to dict, using 2-level MultiIndex
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

# create dataFrame w. dict and dates as index
data = pd.DataFrame(
    index=dates,
    data=data
)

# SORT columns in data 
data = data.reindex(
    sorted(data.columns,
           key=lambda c: c[0]),
    axis=1
)

# REMOVE NANS
data['date'] = data.index
data = data[data.date.apply(lambda d: d.date().isoformat() not in drop_dates)]
del data['date']

data.to_csv(save_path)