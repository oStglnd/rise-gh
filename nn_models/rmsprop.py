
import numpy as np

class RMSProp:
    def __init__(
            self,
            beta: float,
            eps: float,
            weights: list
        ):
        # save init params
        self.beta = beta
        self.eps = eps
        
        # init moments
        self.m = {}
        self.v = {}
        for name, weight in weights.items():
            self.m[name] = np.zeros(weight.shape)
            self.v[name] = np.zeros(weight.shape)
            
    def calcMoment(
            self, 
            moment: np.array, 
            grad: np.array
        ) -> np.array:
        
        new_moment = self.beta * moment + (1 - self.beta) * np.square(grad)
        return new_moment
    
    def step(
            self, 
            weight: str, 
            grad: np.array
        ):     
        
        # update fist moment and correct bias
        self.m[weight] = self.calcMoment(self.m[weight], grad)
        
        step_update = grad / (np.sqrt(self.m[weight]) + self.eps)
        return step_update
