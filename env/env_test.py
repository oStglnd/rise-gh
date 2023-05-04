
import numpy as np

from misc import OUProcess
from greenhouse_v2 import greenhouseEnvironment

# create env
env = greenhouseEnvironment()

# reset env
env.reset()

flowOriginal = env.flowQueue.copy()
tempOriginal = env.tempQueue.copy()

# init params, lists
terminal = False
tempNextList = []
tempNewList = []
rewardList = []

action_0 = flowOriginal[-1][0] + 0.05
ou_process = OUProcess(mu=action_0, sigma=0.05, theta=1.0, dt=1.0)
action = np.array([
    ou_process.sample(), 
    0
])

actionList = [action[0]]

tstep = 0
while not terminal:
    
    state, reward = env.step(action)
    temp_next = state[2]    
    tempNextList.append(temp_next)
    rewardList.append(reward)
    tstep += 1
    
    action = np.array([
        ou_process.sample(),
        0
    ])
    
    actionList.append(action[0])
    
    if tstep % 100 == 0:
        print('STEP {}'.format(tstep))