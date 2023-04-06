
import os
import pandas as pd
import json

from get_data import getData

# path for RISE data
home_path = os.path.dirname(os.getcwd())
data_path = home_path + '\\data\\sensors\\'
save_path = home_path + '\\data\\control.csv'

# get data specs
specs = json.loads(json.load(open(home_path + '\\misc\\specs.json')))
names = specs['control']
drop_dates = specs['dates']

# data categories
data_cat = [
   'flow',
   'state',
   'setpoints'
]

# get old data
data = getData(
    path=data_path+'dec-feb\\',
    cats=data_cat,
    names=names,
    drop_dates=drop_dates
)

# get new data
dataNew = getData(
    path=data_path+'feb-mar\\',
    cats=data_cat,
    names=names,
    drop_dates=drop_dates
)

# concatenate files and save as csv
data = pd.concat((data, dataNew))
data.to_csv(save_path)