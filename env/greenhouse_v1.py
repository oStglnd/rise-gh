
import os
import pandas as pd
import numpy as np

class greenhouseEnvironment():
    
    def __init__(self):
        # define paths
        home_path = os.path.dirname(os.getcwd())
        data_path = home_path + '\\data\\'
        
        # get data
        self.data = pd.read_csv(
            data_path + 'data_env.csv',
            header=[0, 1],
            index_col=[0, 1, 2, 3, 4]
        )
        
        # segment data into days
        dates = zip(
            self.data.index.get_level_values(0),
            self.data.index.get_level_values(1)
        )
        self.dates = pd.Series(dates).unique()
        
        # create epoch-specific subset
        self.epochData = self.data.loc[self.dates[0][0]].loc[self.dates[0][1]]
        
    def reset(self):
        # pick random day
        date = np.random.choice