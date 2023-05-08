
import numpy as np
import matplotlib.pyplot as plt

from misc import OUProcess
from greenhouse_v2 import greenhouseEnvironment

# create env
env = greenhouseEnvironment()

for _ in range(10):
    # reset env
    env.reset()
    
    compData = env.data.loc[env.date][[
        ('state', 'TA01_output'),
        ('state', 'TA02_output'),
        ('temperatures', 'TA01_GT10X_GM10X'),
        ('temperatures', 'TA01_GT401_GM401'),
    ]].values
    
    
    # init params, lists
    terminal = False
    tempNextList = []
    rewardList = []
    
    action_0 = compData[env.t_steps][0] + 0.2
    ou_process = OUProcess(mu=action_0, sigma=0.01, theta=1.0, dt=1.0)
    action = np.array([
        ou_process.sample(), 
        0
    ])
    
    actionList = [action[0]]
    
    tstep = 0
    
    print('\nDATE: {}'.format(env.date))
    while not terminal:
        
        state, reward, terminal = env.step(action)
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
            print('\tSTEP {}'.format(tstep))
            
    
    plt.plot(tempNextList, 'b')
    plt.plot(compData[:len(tempNextList), 2], 'r')
    plt.ylim(15, 35)
    plt.xlim(0, len(tempNextList))
    plt.title('Temperature comparison')
    plt.show()
    
    plt.plot(actionList, 'b')
    plt.plot(compData[:len(actionList), 0], 'r')
    plt.plot(compData[:len(actionList), 1], 'r--')
    plt.ylim(0, 1.0)
    plt.xlim(0, len(actionList))
    plt.title('Action comparison')
    plt.show()
    
    # plt.plot(tempNextList, 'b')
    plt.plot(compData[:len(tempNextList), 3], 'm')
    plt.ylim(0, 1.0)
    plt.xlim(0, len(tempNextList))
    plt.title('TA01 temp')
    plt.show()
