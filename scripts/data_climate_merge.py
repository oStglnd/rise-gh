
import os
import pandas as pd

# define data path
home_path = os.path.dirname(os.getcwd())
data_path = home_path + '\\data\\'
save_path = home_path + '\\data\\climate_merged.csv'

# get sensor data
data = pd.read_csv(
    data_path + 'climate.csv',
    header=[0, 1],
    index_col=0
)
data.index = pd.to_datetime(data.index.values)

# get weather data
weatherdata = pd.read_csv(data_path + 'climate_smhi.csv', index_col=0)
weatherdata.index = pd.to_datetime(weatherdata.index.values)

# add SMHi data to sensors data
# data['temperatures', 'SMHI'] = weatherdata.temp
# data['temperatures', 'SMHI_dewp'] = weatherdata.temp_dewpoint
# data['pressure', 'SMHI'] = weatherdata.pressure
# data['humidity', 'SMHI'] = weatherdata.humidity
data['sun', 'vol'] = weatherdata.sun_vol
data['sun', 'vol_deriv'] = weatherdata.sun_vol_deriv
data['sun', 'vol_raw'] = weatherdata.sun_vol_raw
data['sun', 'gsi'] = weatherdata.sun_gsi
data['sun', 'gsi_deriv'] = weatherdata.sun_gsi_deriv
data['sun', 'gsi_raw'] = weatherdata.sun_gsi_raw
data['wind', 'speed'] = weatherdata.wind_speed
data['wind', 'dir'] = weatherdata.wind_dir
data['wind', 'Wx'] = weatherdata.wind_Wx
data['wind', 'Wy'] = weatherdata.wind_Wy
data['time', 'dayofyear'] = weatherdata.time_doy
data['time', 'monthofyear'] = weatherdata.time_moy
data['time', 'minofday'] = weatherdata.time_mod
data['time', 'minofday_deriv'] = weatherdata.time_mod_deriv
data['time', 'hourofday'] = weatherdata.time_hod

# RE-sort columns by primary key
data = data.reindex(
    sorted(data.columns,
            key=lambda c: c[0]),
    axis=1
)

# save merged data
data.to_csv(save_path)