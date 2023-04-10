
import os
import pandas as pd
import json
import datetime

from get_data import getData

# path for RISE data
home_path = os.path.dirname(os.getcwd())
data_path = home_path + '\\data\\sensors\\'

# get data specs
specs = json.loads(json.load(open(home_path + '\\misc\\specs.json')))
drop_dates = specs['dates']

# data cat dict
catDict = {
    'climate':[
        'humidity',
        'temperatures',
        'pressure'
    ],
    'control':[
        'flow',
        'state',
        'setpoints',
        'power'
    ]}

for cat in catDict:
    # get old data
    data = getData(
        path=data_path+'dec-feb\\',
        cats=catDict[cat],
        names=specs[cat],
        drop_dates=drop_dates
    )
    
    # get new data
    dataNew = getData(
        path=data_path+'feb-mar\\',
        cats=catDict[cat],
        names=specs[cat],
        drop_dates=drop_dates
    )
    
    indxNew = dataNew.index + datetime.timedelta(0,1)
    dataNew.index = indxNew.strftime('%Y-%m-%d %H:%M:%S')#.ceil('s')

    # concatenate files and save as csv
    data = pd.concat((data, dataNew))
    save_path = home_path + '\\data\\{}.csv'.format(cat)
    data.to_csv(save_path)