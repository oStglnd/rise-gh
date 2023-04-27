
import os
import numpy as np

from greenhouse_v1 import greenhouseEnvironment

# create env
env = greenhouseEnvironment()

# reset env
state = env.reset()

# init params, lists
terminal = False
rewardList = []
tempList = []
tempDiffList = []

action = 0.7
tstep = 0
while not terminal:
    
    state, reward, temp, terminal, tempDiff = env.step(action)
    
    rewardList.append(reward)
    tempList.append(temp)
    tempDiffList.append(tempDiff[0][0] / 10)
    
    tstep += 1
    
    if tstep % 100 == 0:
        print('STEP {}'.format(tstep))