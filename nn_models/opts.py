
import numpy as np

class AdaGrad:
    def __init__(
            self,
            eps: float,
            weights: list
        ):
        # save init params
        self.eps = eps
        
        # init dicts for saving moments
        self.m = {}
        
        # init moments
        for name, weight in weights.items():
            self.m[name] = np.zeros(weight.shape)
    
    def calc_moment(
            self,
            moment: np.array,
            grad: np.array
        ) -> np.array:
        
        new_moment = moment + np.square(grad)
        return new_moment
    
    def step(
            self, 
            weight: str, 
            grad: np.array
        ) -> np.array:
        
        self.m[weight] = self.calc_moment(self.m[weight], grad)
        step_update = grad / (np.sqrt(self.m[weight]) + self.eps)
        
        return step_update
    
class Adam:
    def __init__(
            self,
            beta1: float,
            beta2: float,
            eps: float,
            weights: list
        ):
        # save init params
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        
        # init moments
        self.m = {}
        self.v = {}
        for name, weight in weights.items():
            self.m[name] = np.zeros(weight.shape)
            self.v[name] = np.zeros(weight.shape)
            
    def calc_moment(
            self, 
            beta: float, 
            moment: np.array, 
            grad: np.array
        ) -> np.array:
        
        new_moment = beta * moment + (1 - beta) * grad
        return new_moment
    
    def step(
            self, 
            weight: str, 
            grad: np.array, 
            t: int
        ) -> np.array:     
        
        # update fist moment and correct bias
        self.m[weight] = self.calc_moment(
            self.beta1,
            self.m[weight], 
            grad
        )
        
        # update second moment and correct bias
        self.v[weight] = self.calc_moment(
            self.beta2,
            self.v[weight], 
            np.square(grad)
        )
        
        m_corrected = self.m[weight] / (1 - self.beta1 ** t + self.eps)
        v_corrected = self.v[weight] / (1 - self.beta2 ** t + self.eps)
        step_update = m_corrected / (np.sqrt(v_corrected) + self.eps)
        
        return step_update
    
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
            
    def calc_moment(
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
        self.m[weight] = self.calc_moment(self.m[weight], grad)
        
        step_update = grad / (np.sqrt(self.m[weight]) + self.eps)
        return step_update