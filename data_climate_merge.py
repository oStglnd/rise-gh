
import os
import pandas as pd

# define data path
home_path = os.getcwd()
data_path = home_path + '\\data\\'
save_path = home_path + '\\data\\climate_merged.csv'

# get sensor data
data = pd.read_csv(
    data_path + 'climate_sensors.csv',
    header=[0, 1],
    index_col=0
)
data.index = pd.to_datetime(data.index.values)

# get weather data
weatherdata = pd.read_csv(data_path + 'climate_smhi.csv', index_col=0)
weatherdata.index = pd.to_datetime(weatherdata.index.values)

# add SMHi data to sensors data
data['temperatures', 'SMHI'] = weatherdata.temp
data['temperatures', 'SMHI_dewp'] = weatherdata.temp_dewpoint
data['pressure', 'SMHI'] = weatherdata.pressure
data['humidity', 'SMHI'] = weatherdata.humidity
data['sun', 'SMHI_vol'] = weatherdata.sun_vol
data['sun', 'SMHI_gsi'] = weatherdata.sun_gsi
data['wind', 'SMHI_speed'] = weatherdata.wind_speed
data['wind', 'SMHI_dir'] = weatherdata.wind_dir

# "FILL OUT" hourly SMHI data on half-min-freq by interpolating
# using forward fill
data = data.ffill()

# RE-sort columns by primary key
data = data.reindex(
    sorted(data.columns,
            key=lambda c: c[0]),
    axis=1
)

# save merged data
data.to_csv(save_path)