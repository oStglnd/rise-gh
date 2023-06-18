
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
drop_hours = specs['hours']

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

periods = [
    'dec-feb',
    'feb-mar',
    'mar-apr',
    'apr-may'
]

for cat in catDict:
    # get old data
    data = getData(
        path=data_path+periods[0] + '\\',
        cats=catDict[cat],
        names=specs[cat],
        drop_dates=drop_dates,
        drop_hours=drop_hours
    )
    
    for period in periods[1:]:
        # get new data
        dataNew = getData(
            path=data_path+period + '\\',
            cats=catDict[cat],
            names=specs[cat],
            drop_dates=drop_dates,
            drop_hours=drop_hours
        )
        
        indxNew = dataNew.index + datetime.timedelta(0,1)
        dataNew.index = indxNew.strftime('%Y-%m-%d %H:%M:%S')#.floor('s')

        # concatenate files
        data = pd.concat((data, dataNew))
    
    # and save as csv
    data = data[~data.index.duplicated(keep='first')]
    save_path = home_path + '\\data\\{}.csv'.format(cat)
    data.to_csv(save_path)