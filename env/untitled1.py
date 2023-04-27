
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

action = 0.3
step = 0
while not terminal:
    
    state, reward, temp, terminal = env.step(action)
    
    rewardList.append(reward)
    tempList.append(temp)
    
    step += 1
    
    if step % 100:
        print('STEP {}'.format(step))