
import os
import pandas as pd

# path for SMHI data
home_path = os.getcwd()
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
files = os.listdir(data_path)[1:]

# read first instance
data = pd.read_excel(data_path + files[0])
date_fixer(data)

for file in files[1:]:
    newdata = pd.read_excel(data_path + file)
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

# set index
data.set_index(keys='date', inplace=True)

# save data
data.to_csv(save_path)