
import os
import pandas as pd

# define data path
home_path = os.path.dirname(os.getcwd())
data_path = home_path + '\\data\\'
save_path = home_path + '\\data\\data_merged.csv'

# get data for climate variables
data_climate = pd.read_csv(
    data_path + 'climate_merged.csv',
    header=[0, 1],
    index_col=0
)
data_climate.index = pd.to_datetime(data_climate.index.values)

# get data for control variables
data = pd.read_csv(
    data_path + 'control.csv',
    header=[0, 1, 2],
    index_col=0
)
data.index = pd.to_datetime(data.index.values)

# FLATTEN column indices (merge level 1 and 2)
subcols = [
    f'{x}_{y}' for x, y in zip(
        data.columns.get_level_values(level=1), 
        data.columns.get_level_values(level=2)
    )
]

# fix multiIndex columns (merging 1 and 2)
data.columns = pd.MultiIndex.from_tuples(zip(data.columns.get_level_values(level=0), subcols))

# merge datasets
data = data.merge(right=data_climate, left_index=True, right_index=True)

# RE-sort columns by primary key
data = data.reindex(
    sorted(data.columns,
            key=lambda c: c[0]),
    axis=1
)

# save merged data
data.to_csv(save_path)