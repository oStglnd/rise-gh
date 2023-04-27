
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
data.set_index('date', drop=True, inplace=True)
#del data['date']

# new DF w. expanded index
dataExpanded = pd.DataFrame(index=dates)

# get week values
dataExpanded['week'] = dataExpanded.index.get_level_values(0).isocalendar().week
data['week'] = data.index.get_level_values(0).isocalendar().week


# iterate over cols
for col in data.columns:
    
    for week in data.week.unique():
        
        # get subset and size of expanded subset
        subset = data[data.week==week][col].copy()
        
        # get index for expanded data
        idxsExpanded = dataExpanded[dataExpanded.week==week].index
        
        # get xranges
        xRange = np.arange(len(subset))
        xRangeExpanded = np.linspace(0, len(subset), len(idxsExpanded))
    
        # calculate cubic spline and apply to expanded x vals
        spline = CubicSpline(xRange, subset.values)
        colExpanded = spline(xRangeExpanded)
        colExpandedDeriv = spline.derivative()(xRangeExpanded)
        
        # clip values in expanded cols
        colExpanded = np.clip(
            a=colExpanded,
            a_min=subset.min(),
            a_max=subset.max()
        )
        
        # colExpandedDeriv = np.clip(
        #     a=colExpandedDeriv,
        #     # a_min=subset.min(),
        #     # a_max=subset.max()
        # )
        
        # import to expanded DF
        dataExpanded.loc[idxsExpanded, col] = colExpanded
        dataExpanded.loc[idxsExpanded, col + '_deriv'] = colExpandedDeriv

    # shift interpolated series ONE HOUR
    dataExpanded[col] = dataExpanded[col].shift(60).bfill()


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

dataExpanded['time_moy'] = np.sin(
    dataExpanded.index.get_level_values(0).month * (np.pi / 12)
)
dataExpanded['time_hod'] = np.sin(
        dataExpanded.index.get_level_values(0).hour * (np.pi / 24)
)
dataExpanded['time_mod'] = np.sin(
    (dataExpanded.index.get_level_values(0).hour * 60 \
     + dataExpanded.index.get_level_values(0).minute) \
        * (np.pi / (24 * 60))
)
dataExpanded['time_mod_deriv'] = np.cos(
    (dataExpanded.index.get_level_values(0).hour * 60 \
     + dataExpanded.index.get_level_values(0).minute) \
        * (np.pi / (24 * 60))
)
    
# save raw GSI and VOL
dataExpanded['sun_gsi_raw'] = data['sun_gsi']
dataExpanded['sun_vol_raw'] = data['sun_vol']

# fill forward
dataExpanded['sun_gsi_raw'] = dataExpanded['sun_gsi_raw'].ffill()
dataExpanded['sun_vol_raw'] = dataExpanded['sun_vol_raw'].ffill()
    
# save data
dataExpanded.to_csv(save_path)