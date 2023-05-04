
import tensorflow as tf
from keras import layers, Model

class kalmanFilter():
    
    def __init__(self):
        
        # init hyperparams
        self.Q = 1e-5
        self.R = 0.1**3
        
        # init measurements
        self.x = 20
        self.x_pre = 20
        self.p = 1
        self.p_pre = 1
        self.K = 0
        
    def reset(self):
        # reset all measurements
        self.x = 20
        self.x_pre = 20
        self.p = 1
        self.p_pre = 1
        self.K = 0
        
    def update(self, measurement):
        # time update
        self.x_pre = self.x
        self.p_pre = self.p + self.Q
        
        # measurement update
        self.k = self.p_pre / (self.p_pre + self.R)
        self.x = self.x_pre + self.k * (measurement - self.x_pre)
        self.p = (1 - self.k) * self.p_pre
        
        return self.x
