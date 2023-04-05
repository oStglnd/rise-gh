
import os
import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline

# path for SMHI data
home_path = os.path.dirname(os.getcwd())
data_path = home_path + '\\data\\SMHI\\'
save_path = home_path + '\\data\\climate_smhi.csv'

# define DF transformer func
def date_fixer(data):
    data['Datum'] = data['Datum'].apply(lambda d: str(d)[:10])
    data['Tid (UTC)'] = data['Tid (UTC)'].apply(lambda t: str(t))
    data['date'] = pd.to_datetime(data['Datum'] + ' ' + data['Tid (UTC)'])
    
    del data['Datum'], data['Tid (UTC)'], data['Kvalitet']
    try:
        del data['Kvalitet.1']
    except KeyError:
        pass

# iterate over data_path to find files
files = os.listdir(data_path)[2:]

# read first instance
data = pd.read_csv(data_path + files[0], delimiter=';')
date_fixer(data)

for file in files[1:]:
    newdata = pd.read_csv(data_path + file, delimiter=';')
    date_fixer(newdata)
        
    data = pd.merge_ordered(left=data, right=newdata, on='date')
    
# Rename columns
name_dict = {
    'Byvind':'wind_gust', 
    'Daggpunktstemperatur':'temp_dewpoint',
    'Global Irradians (svenska stationer)':'sun_gsi', 
    'Relativ Luftfuktighet':'humidity',
    'Lufttemperatur':'temp', 
    'Lufttryck reducerat havsytans nivå':'pressure',
    'Molnbas (lägsta molnlager)':'cloud_base', 
    'Molnmängd (lägsta molnlager)':'cloud_vol',
    'Total molnmängd':'cloud_tot', 
    'Nederbördsmängd':'rain', 
    'Solskenstid':'sun_vol',
    'Vindriktning':'wind_dir', 
    'Vindhastighet':'wind_speed'
}
data = data.rename(columns=name_dict)

# # set index
# data.set_index(keys='date', inplace=True)

# get EXPANDED date series
dates = pd.date_range(
    start=data.date.iloc[0],
    end=data.date.iloc[-1],
    freq='30s'
)

# interpolate missing vals
data = data.ffill()

# remove date from data
del data['date']

# new DF w. expanded index
dataExpanded = pd.DataFrame(index=dates)

# define x ranges for spline
xRange = np.arange(len(data))
xRangeExpanded = np.linspace(0, len(data), len(dataExpanded))

# iterate over cols
for col in data.columns:
    # calculate cubic spline and apply to expanded x vals
    spline = CubicSpline(xRange, data[col].values)
    colExpanded = spline(xRangeExpanded)
    
    # clip values in expanded col
    colExpanded = np.clip(
        a=colExpanded,
        a_min=data[col].min(),
        a_max=data[col].max()
    )
    
    # import to expanded DF
    dataExpanded[col] = colExpanded

### calculate wind distributions
# Convert wind direction to radians.
windRad = dataExpanded['wind_dir'].values * np.pi / 180

# Calculate the wind x and y components.
dataExpanded['wind_Wx']  = dataExpanded['wind_speed'].values * np.cos(windRad)
dataExpanded['wind_Wy'] = dataExpanded['wind_speed'].values * np.sin(windRad)

# create TIME data, i.e. SIN of DAYS per YEAR and MINUTES per DAY
dataExpanded['time_doy'] = np.sin(
    dataExpanded.index.get_level_values(0).dayofyear * (np.pi / 365)
)
dataExpanded['time_mod'] = np.sin(
    (dataExpanded.index.get_level_values(0).hour * 60 \
     + dataExpanded.index.get_level_values(0).minute) \
        * (np.pi / (24 * 60))
)

# save data
dataExpanded.to_csv(save_path)