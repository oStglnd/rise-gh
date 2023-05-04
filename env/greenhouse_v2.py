
import os
from collections import deque
import pandas as pd
import numpy as np
import tensorflow as tf

from misc import kalmanFilter

class greenhouseEnvironment():
    def __init__(self):
        # define paths
        home_path = os.path.dirname(os.getcwd())
        data_path = home_path + '\\data\\'
        model_path = home_path + '\\model\\saved\\temp_model_v20'
        
        # set env specs
        self.t_steps = 10
        self.n_steps = 30
        
        # set action specs
        self.ta01_min, self.ta02_min = 0.0, 0.0
        self.ta01_max, self.ta02_max = 1.0, 1.0
        
        # set temp specs
        self.temp_min = 10.0
        self.temp_max = 30.0
        
        # get data
        self.data = pd.read_csv(
            data_path + 'data_env.csv',
            header=[0, 1],
            index_col=[0, 1, 2, 3]
        )
        
        # get engine model
        self.model = tf.keras.models.load_model(model_path)
        
        # segment data into days
        dates = zip(
            self.data.index.get_level_values(0),
            self.data.index.get_level_values(1)
        )
        self.dates = pd.Series(dates).unique()
        
        # save current date
        self.date = self.dates[0]
        
        # create epoch-specific subsets f. states
        self.stateInternal = None
        self.stateExternal = None
        
        # create change container for flow var and temp
        self.tempQueue = deque(maxlen=self.t_steps)
        self.flowQueue = deque(maxlen=self.n_steps)
        
        # create container for prev action/flow rate
        self.action_prev = np.zeros(shape=(1,2))
        
        # create container for step and terminal state
        self.tstep = 0
        self.terminal = False
        
        # create kalman filter
        self.kalman = kalmanFilter()
        
    def reset(self):
        # pick random day
        self.date = np.random.choice(self.dates)
        data = self.data.loc[self.date].copy()
        
        # get sequences (w/o. output)
        seqVals = data[[
            ('temperatures', 'DC_GT401_GM401'),
            ('temperatures', 'TA01_GT401_GM401'),
            ('temperatures', 'DC_GT301_damped'),
            ('humidity', 'TA01_GT10X_GM10X'),
            ('sun', 'gsi'),
            ('sun', 'gsi_deriv'),
            ('time', 'minofday')
        ]].values
        
        seqs = []
        for i in range(len(seqVals) - self.n_steps):
            seqs.append(seqVals[i:i+self.n_steps])
        seqs = np.stack(seqs)
        
        # get temperature data
        temps = data.pop(('temperatures', 'TA01_GT10X_GM10X')).values
        temps = temps[self.n_steps-self.t_steps:self.n_steps]
        
        # get flow data
        flows = data[[
            ('state', 'TA01_output'),
            ('state', 'TA02_output')
        ]].values
        flows = flows[:self.n_steps]
        
        # get temperature setpoints
        setpoints = data[('setpoints', 'TA01_GT10X_GM10X')].values
        setpoints = setpoints[self.n_steps-self.t_steps:-self.t_steps]
        
        # create internal states
        self.stateInternal = iter(zip(
            seqs,
            setpoints
        ))
        
        # get external data
        self.stateExternal = iter(data[[
            ('setpoints', 'TA01_GT10X_GM10X'),
            ('temperatures', 'TA01_GT401_GM401'),
            ('temperatures', 'DC_GT301_damped'),
            ('sun', 'gsi'),
            ('sun', 'gsi_deriv'),
            ('time', 'minofday'),
            ('time', 'minofday_deriv'),
        ]].values[self.n_steps-self.t_steps:-self.t_steps])
        
        # fill tempQueue
        for temp in temps:
            self.tempQueue.append(temp)
        
        # fill flowQueue
        for flow in flows:
            self.flowQueue.append(flow)
            
        # reset Kalman filter
        self.kalman.reset(x=temps[-1])
        
        # delete placeholders etc
        del data, seqs, seqVals, setpoints, temps, flows
            
    def step(self, action: np.array):
        
        # clip action to be withing environment bounds
        action = np.clip(
            a=action,
            a_min=[self.ta01_min, self.ta02_min],
            a_max=[self.ta01_max, self.ta02_max]
        )
        
        # put action in flowQueue
        _ = [self.flowQueue.pop() for _ in range(self.t_steps)]
        _ = [self.flowQueue.append(action) for _ in range(self.t_steps)]
        
        # get new state sequence
        state_seq, setpoint = next(self.stateInternal)
        
        # stack flowQueue onto state_seq
        state_seq = np.hstack((
            np.vstack(self.flowQueue),
            state_seq
        ))
        
        # get temperature value
        temp_next = self.tempQueue.popleft()
        
        # predict next temperature value
        temp_new = self.model.predict([
            state_seq[np.newaxis, :].astype('float64'), 
            np.array([temp_next])[np.newaxis, :].astype('float64')
        ], verbose=0)[0][0]
        
        # apply kalman filter and clip
        temp_new = self.kalman.update(temp_new)
        temp_new = np.clip(temp_new, self.temp_min, self.temp_max)
        
        # save in queue
        self.tempQueue.append(temp_new)
        
        # update flowqueue
        self.flowQueue.append(action)
        
        # calculate reward
        reward = -abs(temp_next - setpoint)
        
        # get external state
        state = next(self.stateExternal)
        state = np.hstack((
            action,
            temp_next,
            state
        ))
        
        return state, reward