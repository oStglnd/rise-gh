
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
        self.a_min = 0.0
        self.a_max = 1.0
        
        # set temp specs
        self.tempMin = 10.0
        self.tempMax = 30.0
        
        # get data
        self.data = pd.read_csv(
            data_path + 'data_env.csv',
            header=[0, 1],
            index_col=[0, 1, 2, 3, 4]
        )
        
        # save info on flow outp
        self.flowMin = self.data.state.TA01_output.min()
        self.flowMax = self.data.state.TA01_output.max()
        
        # get engine model
        self.model = tf.keras.models.load_model(model_path)
        
        # segment data into days
        dates = zip(
            self.data.index.get_level_values(0),
            self.data.index.get_level_values(1)
        )
        self.dates = pd.Series(dates).unique()
        
        # create epoch-specific subsets f. states
        self.stateInternal = None
        self.stateExternal = None
        
        # create change container for flow var and temp
        self.tempQueue = deque(maxlen=self.t_steps)
        self.flowQueue = deque(maxlen=self.n_steps)
        
        # create container for prev action/flow rate
        self.action_prev = 0
        
        # create container for step and terminal state
        self.tstep = 0
        self.terminal = False
        
        # create kalman filter
        self.kalman = kalmanFilter()
        
    def reset(self):
        # pick random day
        date = np.random.choice(self.dates)
        data = self.data.loc[date].copy()
        
        # get setpoint vals for reward calc.
        tempSet = data.pop(('setpoints', 'TA01_GT10X_GM10X')).shift(-self.t_steps).values
        
        # get vals for external state
        tempDC = data.pop(('temperatures', 'DC_GT401_GM401')).shift(-self.t_steps).values
        time = data.time.copy().shift(-self.t_steps).values
        del data[('time', 'minofday_deriv')]
        
        # get temp values f. GH and crop w.r.t. n_steps
        tempGH = data.pop(('temperatures', 'TA01_GT10X_GM10X')).values
        tempGH = tempGH[self.n_steps-self.t_steps:]

        # get scaled flow-temp values f. GH
        scaledTempDC = data.pop(('temperatures', 'DC_GT401_GM401_scaled')).values
        
        # get initial flow state
        flowDC = data.pop(('state', 'TA01_output')).shift(-self.t_steps).values
        self.action_prev = (flowDC[self.t_steps] - self.flowMin) / (self.flowMax - self.flowMin)
        
        # get vals for internal state
        vals = data.values
        seqs = []
        for i in range(len(vals) - self.n_steps):
            seqs.append(vals[i:i+self.n_steps])
        seqs = np.stack(seqs)
        
        # fill out the temp queue w. 
        for t in range(self.t_steps):
            self.tempQueue.append(tempGH[t])
        
        # fill out the flow queue w.
        for n in range(self.n_steps):
            self.flowQueue.append(scaledTempDC[n])
        
        # remove superfluous temperature, flow data
        del tempGH, vals, scaledTempDC, flowDC
        
        self.stateExternal = iter(zip(
            tempDC,
            tempSet,
            time,
        ))
        
        self.stateInternal = iter(seqs)
        
        # make state
        state, _, _, _, _ = self.step(self.action_prev)
        
        # reset kalman
        self.kalman.reset()
        
        return state
        
    def step(self, action: float):
        
        # clip action
        action = np.clip(action, self.a_min, self.a_max)
        self.action_prev = action
        
        # rescale action from [0, 1] to [flow.min(), flow.max()] to [1, ...]
        flow = action * (self.flowMax - self.flowMin) + self.flowMin
        flowScale = flow / self.flowMin
        
        # get internal and external states
        stateExternal = next(self.stateExternal)
        stateInternal = next(self.stateInternal)
        
        # recalculate DC-temp-flow
        flowNew = stateExternal[0] * flowScale
        
        # update flowQueue
        for i in range(self.t_steps):
            self.flowQueue[-i] = flowNew
        
        # get temperature
        # temperature = 20.0
        temperature = self.tempQueue.popleft()
        # temperature = np.array([self.tempQueue.popleft()])
        # temperature = np.array([20.0])
        # temperature = [20.0]
        
        # get new sequence block
        seq = np.hstack([
            np.array(self.flowQueue)[:, np.newaxis],
            stateInternal
        ])
        
        # calculate temp diff for t + t_steps
        tempNew = self.model.predict(
            [
                seq[np.newaxis, :].astype('float64'), 
                np.array([temperature])[np.newaxis, :].astype('float64')
            ], 
            verbose=False
        )[0][0]
        
        # apply Kalman filter
        tempNew = self.kalman.update(tempNew)
        
        # set net temp within state range
        tempNew = np.clip(tempNew, self.tempMin, self.tempMax)
        
        # update tempQueue w. new future tempVal
        self.tempQueue.append(tempNew)
        
        # shift flowQueue
        # self.flowQueue.append(flowNew)
        
        # calculate reward (current temperature - setpoint)
        reward = np.abs(temperature - stateExternal[1])
        
        # make state
        state = np.stack([
            temperature,            # GH temperature
            stateExternal[0],       # DC temperature
            stateExternal[1],       # GH temperature setpoint
            stateExternal[-1][0],   # minute of day
            stateExternal[-1][1],   # minute of day, derivative
            self.action_prev,       # previous action, current flow
        ])
        
        # update step and terminal
        self.tstep += 1
        if self.tstep >= 2880:
            self.terminal = True
        
        return state, reward, temperature, self.terminal, tempNew
        
    